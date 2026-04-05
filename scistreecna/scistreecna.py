import warnings
from contextlib import nullcontext
import scistree2 as s2
import numpy as np
import cupy as cp
from time import time
from rich.console import Console
from rich import print
from . import util, external
from .cn_estimate import *
from .topological_sort import *
from .transition_solver import *
from .base.tree import struct_copy_tree


console = Console(force_jupyter=False, log_path=False, width=96)
console.is_jupyter = False
warnings.filterwarnings("ignore")  # opts on log 0 is normal, a -inf is always expected.
cp.set_printoptions(suppress=True)


def set_cuda_device(gpu_id: int = 0) -> None:
    num_gpus = cp.cuda.runtime.getDeviceCount()
    assert gpu_id < num_gpus, f"Error: GPU {gpu_id} is not available."
    cp.cuda.Device(gpu_id).use()


def estimate_batch_sizes(
    n_cells: int,
    n_sites: int,
    cn_min: int = 1,
    cn_max: int = 5,
    gpu_id: int = 0,
    mem_fraction: float = 0.75,
) -> dict:
    """Estimate optimal tree_batch_size given problem size and GPU memory.

    Memory model for marginal_evaluate_dp_batch:
        4 contiguous arrays (all_U, all_U_, all__U, all_Q), each of shape
        (total_nodes, n_sites, N) float32, where:
            total_nodes = tree_batch_size * (2 * n_cells - 1)
            N = (cn_max - cn_min + 1) * (cn_max + cn_min + 2) / 2
        plus probs array: (n_cells, n_sites, N) float32

    Args:
        n_cells:  Number of cells (leaves).
        n_sites:  Number of loci/sites.
        cn_min:   Minimum copy number.
        cn_max:   Maximum copy number.
        gpu_id:   GPU device id.
        mem_fraction: Fraction of free GPU memory to use (default 0.75).

    Returns:
        dict with tree_batch_size, node_batch_size (deprecated, kept for compat),
        and diagnostic info.
    """
    N = int((cn_max - cn_min + 1) * (cn_max + cn_min + 2) / 2)
    nodes_per_tree = 2 * n_cells - 1

    # GPU memory
    free_mem, total_mem = cp.cuda.Device(gpu_id).mem_info
    usable = int(free_mem * mem_fraction)

    # Fixed cost: probs array
    probs_bytes = n_cells * n_sites * N * 4
    # Fixed cost: transition matrices, CuPy overhead, etc.
    overhead = 256 * 1024 * 1024  # ~256MB conservative estimate
    available = usable - probs_bytes - overhead

    # Per-tree cost: 4 arrays × nodes_per_tree × n_sites × N × 4 bytes
    bytes_per_tree = 4 * nodes_per_tree * n_sites * N * 4

    tree_batch_size = max(1, available // bytes_per_tree)

    # Clamp: no point exceeding total NNI candidates (~4 * n_internal_nodes)
    max_candidates = 4 * (n_cells - 1) + 1
    tree_batch_size = min(tree_batch_size, max_candidates)

    # Also respect CUDA grid z-limit (65535) for the leaf layer
    max_by_grid = 65535 // n_cells
    if tree_batch_size > max_by_grid:
        tree_batch_size = max_by_grid

    info = {
        "tree_batch_size": tree_batch_size,
        "node_batch_size": tree_batch_size,  # kept for API compat, not used in optimized path
        "N_states": N,
        "nodes_per_tree": nodes_per_tree,
        "bytes_per_tree_MB": bytes_per_tree / 1024 / 1024,
        "gpu_free_MB": free_mem / 1024 / 1024,
        "gpu_total_MB": total_mem / 1024 / 1024,
        "estimated_mem_usage_MB": (probs_bytes + tree_batch_size * bytes_per_tree) / 1024 / 1024,
    }
    return info


class NodeBatchLoader:
    def __init__(self, trees, batch_size):
        self.trees = trees
        self.batch_size = batch_size  # number of trees
        self.traversor = util.TraversalGenerator()
        self.orders = ["up", "down", "all"]

    def get_all_nodes(self):
        nodes = []
        for tid, tree in enumerate(self.trees):
            ns = tree.get_all_nodes().values()
            for n in ns:
                n.tid = tid  # add tree id into node
            nodes += ns
        return nodes

    def __len__(self):
        n = len(self.trees[0].get_all_nodes())
        return len(self.trees) * n

    def __call__(self, order="up"):
        assert order in self.orders, "invalid order!"
        if order == "all":
            nodes = self.get_all_nodes()
            num_batch = (len(self) + self.batch_size - 1) // self.batch_size
            for b in range(num_batch):
                yield nodes[b * self.batch_size : (b + 1) * self.batch_size]
        else:
            layers = batch_topological_sort(self.trees, order=order)
            for layer in layers:
                num_batch = (len(layer) + self.batch_size - 1) // self.batch_size
                for b in range(num_batch):
                    yield layer[b * self.batch_size : (b + 1) * self.batch_size]


class TreeBatchLoader:
    def __init__(self, trees, batch_size=128):
        self.trees = trees
        self.batch_size = batch_size

    def __len__(self):
        return len(self.trees)

    def __call__(self):
        num_batch = (len(self) + self.batch_size - 1) // self.batch_size
        for b in range(num_batch):
            yield self.trees[b * self.batch_size : (b + 1) * self.batch_size]


class ScisTreeCNA:
    def __init__(
        self, CN_MAX=3, CN_MIN=1, LAMBDA_C=1, LAMBDA_S=1, LAMBDA_T=50, verbose=True
    ):
        self.CN_MAX = CN_MAX
        self.CN_MIN = CN_MIN
        self.LAMBDA_C = LAMBDA_C
        self.LAMBDA_S = LAMBDA_S
        self.LAMBDA_T = LAMBDA_T
        self.N = int((CN_MAX - CN_MIN + 1) * (CN_MAX + CN_MIN + 2) / 2)
        self._indexing()
        transition = TransitionProbability(
            CN_MAX=self.CN_MAX,
            CN_MIN=self.CN_MIN,
            LAMBDA_C=self.LAMBDA_C,
            LAMBDA_S=self.LAMBDA_S,
            LAMBDA_T=self.LAMBDA_T,
        )
        self.tran_prob_mutation_free = cp.log(
            cp.asarray(transition.solve_no_mutation(verbose=False), dtype=cp.float32)
        )
        self.tran_prob_mutation = cp.log(
            cp.asarray(transition.solve_mutation(verbose=False), dtype=cp.float32)
        )
        self.traversor = util.TraversalGenerator()
        if verbose:
            print("with mutation")
            print(transition.format(self.tran_prob_mutation.get()))
            print("mutation free")
            print(transition.format(self.tran_prob_mutation_free.get()))
        # print(self.tran_prob_mutation_free)
        # print(self.tran_prob_mutation)
        # print(self.tran_prob_mutation.shape)
        # print(self.tran_prob_mutation_free.shape)

    def valid(self, g0, g1):
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def _indexing(self):
        state2index = {}
        index2state = {}
        for i in range(0, self.CN_MAX + 1):
            for j in range(0, self.CN_MAX + 1):
                if self.valid(i, j):
                    index = int(
                        (i + j) * (i + j + 1) / 2
                        + i
                        - int((self.CN_MIN) * (self.CN_MIN + 1) / 2)
                    )
                    state2index[(i, j)] = index
                    index2state[index] = (i, j)
        self.state2index = state2index
        self.index2state = index2state
        # print(self.state2index)

    # def index_gt(self, i, j):
    #     return int((i+j) * (i+j+1) / 2 + i - int((self.CN_MIN) * (self.CN_MIN + 1) / 2))

    def index_gt(self, i, j):
        """
        Index for state
        """
        return self.state2index[(i, j)]

    def cn_profile_at_index(self, index):
        """
        CN profie at index
        """
        return self.index2state[index]

    def index(self, i, j, p, q):
        """
        Index for transition matrix
        """
        # use yang's tri indexing
        i1 = self.index_gt(i, j)
        i2 = self.index_gt(p, q)
        return int(i1 * self.N + i2)

    def preprocess_reads_with_missing_values(self, reads):
        cn = reads[:, :, 2]
        mask = cn == -1  # -1 indicates a missing
        # reads[:, :, 2][mask] = -1
        return reads, mask

    def init_prob_leaves_gpu(self, reads, ado=0.1, seqerr=0.001, cnerr=0.2, af=None):
        # here assume copy number is always correct.
        reads = cp.asarray(reads, dtype=np.float32)
        reads, mask = self.preprocess_reads_with_missing_values(reads)
        nsite, ncell = reads.shape[:2]
        ref = reads[:, :, 0]
        alt = reads[:, :, 1]
        cn = reads[:, :, 2]
        # Estimate allele frequencies
        if af is None:
            raf = cp.sum(ref / (cn / 2), axis=1)
            aaf = cp.sum(alt / (cn / 2), axis=1)
            afs = raf / (raf + aaf)
        else:
            afs = cp.ones(nsite, dtype=cp.float32) * af
        N = int((self.CN_MAX - self.CN_MIN + 1) * (self.CN_MAX + self.CN_MIN + 2) / 2)
        probs = cp.zeros((ncell * nsite, N), dtype=cp.float32)
        afs = cp.array(afs, dtype=cp.float32)
        threads_per_block = 128
        blocks_per_grid = (nsite * ncell + threads_per_block - 1) // threads_per_block
        # print('threads per block', threads_per_block, 'blocks per grid', blocks_per_grid)
        # compute_genotype_log_probs()((blocks_per_grid,), (threads_per_block,), (
        #     ref.ravel(), alt.ravel(), cn.ravel(), afs, probs.ravel(),
        #     np.float32(ado), np.float32(seqerr),
        #     np.int32(ncell), np.int32(nsite),
        #     np.int32(self.CN_MAX), np.int32(self.CN_MIN), np.int32(N)
        # ))
        # compute_genotype_log_probs_cn_noise_origin()((blocks_per_grid,), (threads_per_block,), (
        #     ref.ravel(), alt.ravel(), cn.ravel(), afs, probs.ravel(),
        #     np.float32(ado), np.float32(seqerr), np.float32(cnerr),
        #     np.int32(ncell), np.int32(nsite),
        #     np.int32(self.CN_MAX), np.int32(self.CN_MIN), np.int32(N)
        # ))
        util.compute_genotype_log_probs_cn_noise()(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                ref.ravel(),
                alt.ravel(),
                cn.ravel(),
                afs,
                probs.ravel(),
                np.float32(ado),
                np.float32(seqerr),
                np.float32(cnerr),
                np.int32(ncell),
                np.int32(nsite),
                np.int32(self.CN_MAX),
                np.int32(self.CN_MIN),
                np.int32(N),
            ),
        )
        probs = cp.ascontiguousarray(
            cp.transpose(probs.reshape((nsite, ncell, N)), (1, 0, 2))
        )
        # mask = mask.T # (nsite, ncell) -> (ncell, nsite)
        return probs

    def pairwise_distance_matrix(self, probs):
        ncell, nsite, num_states = probs.shape
        exp_probs = cp.exp(probs)
        expected_distances = cp.einsum("ipq,jpq->ij", exp_probs, 1 - exp_probs)
        # print(expected_distances)
        return expected_distances

    def initial_tree(self, probs):
        """
        Build the initial tree by NJ
        """
        distance = self.pairwise_distance_matrix(probs)
        tree = util.neighbor_joining(distance)
        # print(sorted(tree.get_leaves()))
        return tree

    _MAX_GRID_Z = 65535  # CUDA grid z-dimension limit

    def _get_block_grid(self, h, w, batch):
        """Compute optimal block/grid sizes for (h, w) matrices with batch."""
        block_y = min(32, w)
        block_x = max(1, min(256 // block_y, 32))
        block_size = (block_x, block_y)
        grid_size = (
            (h + block_size[0] - 1) // block_size[0],
            (w + block_size[1] - 1) // block_size[1],
            batch,
        )
        return block_size, grid_size

    def _launch_batched_matmul(self, kernel, ptr_src, mat2, ptr_dst, nb, h, w, shared_mem):
        """Launch a batch kernel, automatically splitting if nb > MAX_GRID_Z."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_size, grid_size = self._get_block_grid(h, w, chunk)
            kernel(grid_size, block_size,
                   (ptr_src[off:off+chunk], mat2, ptr_dst[off:off+chunk], chunk, h, w),
                   shared_mem=shared_mem)

    def _launch_batched_add_matmul(self, kernel, ptr_a, ptr_b, mat2, ptr_dst, nb, h, w, shared_mem):
        """Launch fused add+matmul kernel, splitting if nb > MAX_GRID_Z."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_size, grid_size = self._get_block_grid(h, w, chunk)
            kernel(grid_size, block_size,
                   (ptr_a[off:off+chunk], ptr_b[off:off+chunk], mat2, ptr_dst[off:off+chunk],
                    chunk, h, w),
                   shared_mem=shared_mem)

    def _launch_batched_matadd(self, kernel, ptr_a, ptr_b, ptr_dst, nb, h, w):
        """Launch batch matadd kernel, splitting if nb > MAX_GRID_Z."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_size, grid_size = self._get_block_grid(h, w, chunk)
            kernel(grid_size, block_size,
                   (ptr_a[off:off+chunk], ptr_b[off:off+chunk], ptr_dst[off:off+chunk],
                    chunk, h, w))

    def _launch_batched_3vecdot(self, kernel, ptr_a, ptr_b, ptr_c, out, nb, h, w):
        """Launch batch 3vecdot kernel, splitting if nb > MAX_GRID_Z."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_v = (256, 1)
            grid_v = ((h + 255) // 256, 1, chunk)
            kernel(grid_v, block_v,
                   (ptr_a[off:off+chunk], ptr_b[off:off+chunk], ptr_c[off:off+chunk],
                    out[off:off+chunk], chunk, h, w))

    def marginal_evaluate_dp_batch(self, probs, trees, batch_size=512):
        """Evaluate all trees in batch using contiguous GPU arrays + index-based pointers.
        Eliminates Python-loop bottleneck from the old per-node attribute assignment."""
        h, w = probs.shape[1:]
        shared_mem = w * w * 4
        stride = h * w * 4  # bytes per (h, w) float32 slice

        # ====== Phase 1: Pre-compute topology (one-time Python work) ======
        node_id_map = {}  # id(node) -> flat index
        idx = 0
        for tid, tree in enumerate(trees):
            nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
            for nid, node in nodes_dict.items():
                node_id_map[id(node)] = idx
                node.tid = tid
                idx += 1
        total_nodes = idx
        _get = node_id_map.__getitem__

        # Build topological layers with pre-computed GPU index arrays
        layers_up = batch_topological_sort(trees, order="up")
        up_layers = []
        for layer in layers_up:
            leaf_idx = []
            leaf_cells = []
            int_idx = []
            int_c0 = []
            int_c1 = []
            for n in layer:
                if n.is_leaf():
                    leaf_idx.append(_get(id(n)))
                    leaf_cells.append(int(n.name))
                else:
                    int_idx.append(_get(id(n)))
                    ch = n._children
                    int_c0.append(_get(id(ch[0])))
                    int_c1.append(_get(id(ch[1])))
            if leaf_idx:
                up_layers.append(('leaf',
                    cp.array(leaf_idx, dtype=cp.int64),
                    cp.array(leaf_cells, dtype=cp.int64)))
            if int_idx:
                up_layers.append(('internal',
                    cp.array(int_idx, dtype=cp.int64),
                    cp.array(int_c0, dtype=cp.int64),
                    cp.array(int_c1, dtype=cp.int64)))

        layers_down = batch_topological_sort(trees, order="down")
        down_layers = []
        for layer in layers_down:
            has_root = False
            nr_idx = []
            nr_par = []
            nr_sib = []
            for n in layer:
                if n.is_root():
                    has_root = True
                else:
                    nr_idx.append(_get(id(n)))
                    nr_par.append(_get(id(n.parent)))
                    nr_sib.append(_get(id(n.get_siblings()[0])))
            if has_root:
                down_layers.append(('root',))
            if nr_idx:
                down_layers.append(('internal',
                    cp.array(nr_idx, dtype=cp.int64),
                    cp.array(nr_par, dtype=cp.int64),
                    cp.array(nr_sib, dtype=cp.int64)))

        # Pre-compute scoring indices (all nodes, grouped by tree)
        nr_self_l = []
        nr_sib_l = []
        nr_par_l = []
        root_list = []
        for tid, tree in enumerate(trees):
            nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
            for nid, node in nodes_dict.items():
                nidx = _get(id(node))
                if node.is_root():
                    root_list.append(nidx)
                else:
                    nr_self_l.append(nidx)
                    nr_sib_l.append(_get(id(node.get_siblings()[0])))
                    nr_par_l.append(_get(id(node.parent)))
        nr_self_gpu = cp.array(nr_self_l, dtype=cp.int64)
        nr_sib_gpu = cp.array(nr_sib_l, dtype=cp.int64)
        nr_par_gpu = cp.array(nr_par_l, dtype=cp.int64)
        root_gpu = cp.array(root_list, dtype=cp.int64)

        # ====== Phase 2: Allocate contiguous GPU arrays (single allocation) ======
        _buf3 = cp.zeros((3 * total_nodes, h, w), dtype=cp.float32)
        all_U  = _buf3[0*total_nodes:1*total_nodes]
        all_U_ = _buf3[1*total_nodes:2*total_nodes]
        all__U = _buf3[2*total_nodes:3*total_nodes]
        base_U  = all_U.data.ptr
        base_U_ = all_U_.data.ptr
        base__U = all__U.data.ptr

        # ====== Phase 3: Bottom-up (U) pass ======
        _matmul = util.batch_log_matmul_cuda()
        _add_matmul = util.batch_add_log_matmul_cuda()
        _matadd = util.batch_matadd_cuda()
        _3vecdot = util.batch_log_3vecdot_cuda()

        for layer_info in up_layers:
            if layer_info[0] == 'leaf':
                _, indices, cell_ids = layer_info
                nb = len(indices)
                all_U[indices] = probs[cell_ids]
                ptr_u  = base_U  + indices * stride
                ptr_u_ = base_U_ + indices * stride
                ptr__u = base__U + indices * stride
                self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
                self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)
            else:
                _, indices, c0, c1 = layer_info
                nb = len(indices)
                ptr_c0_u_ = base_U_ + c0 * stride
                ptr_c1_u_ = base_U_ + c1 * stride
                ptr_u  = base_U  + indices * stride
                ptr_u_ = base_U_ + indices * stride
                ptr__u = base__U + indices * stride
                self._launch_batched_add_matmul(_add_matmul, ptr_c0_u_, ptr_c1_u_, self.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
                self._launch_batched_matadd(_matadd, ptr_c0_u_, ptr_c1_u_, ptr_u, nb, h, w)
                self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)

        # ====== Phase 4: Top-down (Q) pass ======
        all_Q = cp.zeros((total_nodes, h, w), dtype=cp.float32)
        all_Q[:, :, self.index_gt(2, 0)] = 1.0
        all_Q = cp.log(all_Q)
        base_Q = all_Q.data.ptr
        tran_prob_mutation_free_t = cp.ascontiguousarray(self.tran_prob_mutation_free.T)

        for layer_info in down_layers:
            if layer_info[0] == 'root':
                pass
            else:
                _, indices, par, sib = layer_info
                nb = len(indices)
                ptr_par_q  = base_Q  + par * stride
                ptr_sib_u_ = base_U_ + sib * stride
                ptr_q      = base_Q  + indices * stride
                self._launch_batched_add_matmul(_add_matmul, ptr_par_q, ptr_sib_u_, tran_prob_mutation_free_t, ptr_q, nb, h, w, shared_mem)

        # ====== Phase 5: Scoring ======
        n_nonroot = len(nr_self_gpu)
        n_root = len(root_gpu)

        # Non-root: 3vecdot(_U[self], U_[sib], Q[par])
        likelihoods = cp.log(cp.zeros((n_nonroot, h), dtype=cp.float32))
        ptr__u     = base__U + nr_self_gpu * stride
        ptr_sib_u_ = base_U_ + nr_sib_gpu * stride
        ptr_par_q  = base_Q  + nr_par_gpu * stride
        self._launch_batched_3vecdot(_3vecdot, ptr__u, ptr_sib_u_, ptr_par_q, likelihoods, n_nonroot, h, w)

        # Root: extract U at specific genotype states
        root_U = all_U[root_gpu]  # (n_root, h, w)
        likelihoods_root = cp.zeros((n_root, 3, h), dtype=cp.float32)
        likelihoods_root[:, 0, :] = root_U[:, :, self.index_gt(0, 2)]
        likelihoods_root[:, 1, :] = root_U[:, :, self.index_gt(1, 1)]
        likelihoods_root[:, 2, :] = root_U[:, :, self.index_gt(2, 0)]
        likelihoods_root = likelihoods_root.reshape(n_root * 3, h)

        # Combine and reduce
        likelihoods = likelihoods.reshape(len(trees), -1, h)
        likelihoods = cp.concatenate(
            [likelihoods, likelihoods_root.reshape(len(trees), -1, h)], axis=1
        )
        likelihoods = likelihoods.max(axis=1).sum(axis=-1, dtype=cp.float64)
        return likelihoods

    def calcualte_U(self, tree, probs):
        """
        Bottom-up
        """
        h, w = probs.shape[1:]
        block_y = min(32, w)
        block_x = max(1, min(256 // block_y, 32))
        block_size = (block_x, block_y)
        grid_size = (
            (h + block_size[0] - 1) // block_size[0],
            (w + block_size[1] - 1) // block_size[1],
        )
        shared_mem = w * w * 4  # sizeof(float) * k * k
        ts = 0
        for node in self.traversor(tree):
            if node.is_leaf():
                node.U = probs[int(node.name)].astype(cp.float32)
                node.U_ = cp.zeros([h, w], dtype=cp.float32)
                node._U = cp.zeros([h, w], dtype=cp.float32)
                util.log_matmul_cuda()(
                    grid_size, block_size,
                    (node.U, self.tran_prob_mutation_free, node.U_, h, w),
                    shared_mem=shared_mem,
                )
                util.log_matmul_cuda()(
                    grid_size, block_size,
                    (node.U, self.tran_prob_mutation, node._U, h, w),
                    shared_mem=shared_mem,
                )
                continue
            components = []
            for child in node.get_children():
                components.append(child.U_)
            components = cp.array(components)
            node.U = cp.sum(components, axis=0)
            node.U_ = cp.zeros([h, w], dtype=cp.float32)
            node._U = cp.zeros([h, w], dtype=cp.float32)
            t = time()
            util.log_matmul_cuda()(
                grid_size, block_size,
                (node.U, self.tran_prob_mutation_free, node.U_, h, w),
                shared_mem=shared_mem,
            )
            util.log_matmul_cuda()(
                grid_size, block_size,
                (node.U, self.tran_prob_mutation, node._U, h, w),
                shared_mem=shared_mem,
            )
        return ts

    def calculate_Q(self, tree):
        """
        calcuate Q for each site recursively, have to do after calculation of U, use this fashion instead of DFS
        """
        assert "U" in tree.root.__dict__, "fatal: calculate U first!"
        nsite = tree.root.U.shape[0]
        tran_prob_mut_free_broadcast = cp.tile(
            self.tran_prob_mutation_free, (nsite, 1)
        ).reshape(nsite, self.N, self.N)
        for node in self.traversor(tree, order="pre"):
            if node.is_root():
                ident = cp.log(cp.identity(self.N)).astype(cp.float32)
                node.Q = cp.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
            else:
                # node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
                node.Q = node.parent.Q.copy()
                for sib in node.get_siblings():
                    node.Q += sib.U_.reshape(nsite, 1, self.N)
                node.Q = util.log_matmul(node.Q, tran_prob_mut_free_broadcast)

    def marginal_evaluate_dp(self, probs, tree):
        """
        DP speedup
        """
        nsite = probs.shape[1]
        self.calcualte_U(tree, probs)
        # self.calculate_Q(tree.root)
        self.calculate_Q(tree)
        # loop all branches to place mutation
        likelihoods = []
        branches = []
        for node in self.traversor(tree):
            if node.is_root():
                likelihoods.append(node.U[:, self.index_gt(1, 1)])
                likelihoods.append(node.U[:, self.index_gt(0, 2)])
                likelihoods.append(node.U[:, self.index_gt(2, 0)])
                # print(node.name, likelihood)
                continue
            res = node._U
            for sib in node.get_siblings():
                res += sib.U_
            likelihood = util.log_mat_vec_mul(
                node.parent.Q, res.reshape(nsite, 1, self.N)
            )
            likelihoods.append(likelihood[:, self.index_gt(2, 0)])
            branches.append(node.identifier)
        likelihoods = cp.array(likelihoods)
        # print('ll', likelihoods)
        indicies = cp.argmax(likelihoods, axis=0)
        max_L = likelihoods.max(axis=0).sum()
        return max_L, indicies

    # def _bfs(self, node, site_index, state_index):
    #     cn_profile = self.cn_profile_at_index(state_index)
    #     node.cn = f'{sum(cn_profile)}:{cn_profile}'
    #     if not node.is_leaf():
    #         for child in node.get_children():
    #             child_state_index = node.arg[child.identifier][site_index][state_index]
    #             self._bfs(child, site_index, child_state_index)
    #     else:
    #         node.cn = f'{node.name} [CN {node.cn}] [READ:{node.reads[site_index]}]'

    def nni_search(self, tree):
        """
        NNI neighbor
        """
        tree.draw()
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf()
            for child in tree[node].get_children():
                if child.is_leaf():
                    switch = False
            if switch:
                lc1, lc2 = tree[node].get_children()[0].get_children()
                rc1, rc2 = tree[node].get_children()[1].get_children()

    def nni_search_sinlge_round_batch(
        self, probs, tree, tree_batch_size=64, node_batch_size=32
    ):
        """
        NNI neighbor
        """
        # assert not hasattr(tree.root, 'U'), "tree is not empty."
        candidates = []
        # TODO: quartet switch
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf()
            for child in tree[node].get_children():
                if child.is_leaf():
                    switch = False
            if switch:
                t1 = struct_copy_tree(tree)
                p1 = t1[node].get_children()[0]
                p2 = t1[node].get_children()[1]
                lc1, lc2 = p1.get_children()
                rc1, rc2 = p2.get_children()
                p1.remove_child(lc1)
                p2.remove_child(rc1)
                lc1.set_parent(p2)
                rc1.set_parent(p1)
                p1.add_child(rc1)
                p2.add_child(lc1)
                candidates.append(t1)
                t2 = struct_copy_tree(tree)
                p1 = t2[node].get_children()[0]
                p2 = t2[node].get_children()[1]
                lc1, lc2 = p1.get_children()
                rc1, rc2 = p2.get_children()
                p1.remove_child(lc1)
                p2.remove_child(rc2)
                lc1.set_parent(p2)
                rc2.set_parent(p1)
                p1.add_child(rc2)
                p2.add_child(lc1)
                candidates.append(t2)
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf() and not tree[node].is_root()
            if switch:
                t1 = struct_copy_tree(tree)
                sib = t1[node].get_siblings()[0]
                c1 = t1[node].get_children()[0]
                c2 = t1[node].get_children()[1]
                t1[node].remove_child(c1)
                t1[node].parent.remove_child(sib)
                t1[node].add_child(sib)
                sib.set_parent(t1[node])
                t1[node].parent.add_child(c1)
                c1.set_parent(t1[node].parent)
                candidates.append(t1)
                t2 = struct_copy_tree(tree)
                sib = t2[node].get_siblings()[0]
                c1 = t2[node].get_children()[0]
                c2 = t2[node].get_children()[1]
                t2[node].remove_child(c2)
                t2[node].parent.remove_child(sib)
                t2[node].add_child(sib)
                sib.set_parent(t2[node])
                t2[node].parent.add_child(c2)
                c2.set_parent(t2[node].parent)
                candidates.append(t2)
        # local search: include base tree in the first batch to avoid separate single-tree eval
        best_tree = struct_copy_tree(tree)
        all_trees = [best_tree] + candidates  # base tree at index 0
        loader = TreeBatchLoader(all_trees, batch_size=tree_batch_size)
        best_likelihood = cp.float64(-np.inf)
        num_tree_evaulated = 0
        for bi, trees in enumerate(loader()):
            likelihoods = self.marginal_evaluate_dp_batch(
                probs, trees, batch_size=node_batch_size
            )
            num_tree_evaulated += len(trees)
            max_idx = cp.argmax(likelihoods)
            max_lh = likelihoods[max_idx]
            if max_lh > best_likelihood:
                best_likelihood = max_lh
                best_tree = all_trees[int(bi * tree_batch_size + max_idx)]
        return best_tree, best_likelihood

    def nni_search_non_optim_sinlge_round(self, probs, tree):
        """
        NNI neighbor
        """
        candidates = []
        # tree.draw()
        # TODO: quartet switch
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf()
            for child in tree[node].get_children():
                if child.is_leaf():
                    switch = False
            if switch:
                t1 = tree.copy()
                p1 = t1[node].get_children()[0]
                p2 = t1[node].get_children()[1]
                lc1, lc2 = p1.get_children()
                rc1, rc2 = p2.get_children()
                p1.remove_child(lc1)
                p2.remove_child(rc1)
                lc1.set_parent(p2)
                rc1.set_parent(p1)
                p1.add_child(rc1)
                p2.add_child(lc1)
                candidates.append(t1)
                t2 = tree.copy()
                p1 = t2[node].get_children()[0]
                p2 = t2[node].get_children()[1]
                lc1, lc2 = p1.get_children()
                rc1, rc2 = p2.get_children()
                p1.remove_child(lc1)
                p2.remove_child(rc2)
                lc1.set_parent(p2)
                rc2.set_parent(p1)
                p1.add_child(rc2)
                p2.add_child(lc1)
                candidates.append(t2)
        # TODO: triplet switch
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf() and not tree[node].is_root()
            if switch:
                t1 = tree.copy()
                sib = t1[node].get_siblings()[0]
                c1 = t1[node].get_children()[0]
                c2 = t1[node].get_children()[1]
                t1[node].remove_child(c1)
                t1[node].parent.remove_child(sib)
                t1[node].add_child(sib)
                sib.set_parent(t1[node])
                t1[node].parent.add_child(c1)
                c1.set_parent(t1[node].parent)
                candidates.append(t1)
                t2 = tree.copy()
                sib = t2[node].get_siblings()[0]
                c1 = t2[node].get_children()[0]
                c2 = t2[node].get_children()[1]
                t2[node].remove_child(c2)
                t2[node].parent.remove_child(sib)
                t2[node].add_child(sib)
                sib.set_parent(t2[node])
                t2[node].parent.add_child(c2)
                c2.set_parent(t2[node].parent)
                candidates.append(t2)
        # local search:
        best_tree = tree.copy()
        best_likelihood = self.marginal_evaluate_dp(probs, best_tree)
        for t in candidates:
            # print('spr', popgen.utils.spr_distance(tree, t))
            likelihood = self.marginal_evaluate_dp(probs, t.copy())
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_tree = t
        return best_tree, best_likelihood

    def local_search_batch(
        self,
        probs,
        tree,
        max_iter=0,
        ground_truth=None,
        tree_batch_size=64,
        node_batch_size=32,
        verbose=True,
        verbose_mode="all",
    ):
        # tree = self.initial_tree(probs)
        assert verbose and verbose_mode in [
            "all",
            "min",
        ], "verbose mode should be set to either 'all' or 'min'."
        L = -np.inf
        max_iter = np.inf if max_iter == 0 else max_iter
        iters = 0
        is_converge = False
        context = (
            console.status("[bold green]NNI Searching") if verbose else nullcontext()
        )
        with context:
            while iters < max_iter:
                better_tree, likelihood = self.nni_search_sinlge_round_batch(
                    probs,
                    tree,
                    tree_batch_size=tree_batch_size,
                    node_batch_size=node_batch_size,
                )
                if likelihood <= L:
                    is_converge = True
                    break
                else:
                    L = likelihood
                    tree = better_tree
                    str_log = f"\r[Iteration {iters}]\tLikelihood: {L:.4f}"
                    if ground_truth is not None and isinstance(
                        ground_truth, util.BaseTree
                    ):
                        str_log += f"\tTree accuracy: {util.tree_accuracy(ground_truth, tree):.4f}"
                        str_log += f"\tnRF: {util.normalized_rf_distance(ground_truth, tree):.4f}"
                    if verbose:
                        if verbose_mode == "all":
                            console.log(str_log)
                        else:
                            context.update(
                                f"[bold green]NNI Searching[/bold green]\t{str_log}"
                            )
                    iters += 1
            if verbose:
                if is_converge:
                    console.log(
                        f"[bold red]Local search converge. Best Likelihood: {L}"
                    )
                else:
                    console.log(
                        f"[bold red]Maximal iterations reached. Best Likelihood: {L}"
                    )
        return tree, L

    def local_search(self, probs, tree, ground_truth=None):
        # tree = self.initial_tree(probs)
        L = -np.inf
        while True:
            better_tree, likelihood = self.nni_search_non_optim_sinlge_round(
                probs, tree
            )
            if likelihood <= L:
                print("converge, stop!")
                break
            else:
                L = likelihood
                tree = better_tree
                print("new tree evaluated", L, end=" ")
                if ground_truth:
                    print("acc:", util.tree_accuracy(ground_truth, tree))
        return tree, L

    def maximal_evaluate(
        self, probs, tree, return_trees=False, reads=None, masks=None, use_gpu=True
    ):
        xp = cp if use_gpu else np
        if not use_gpu:
            probs = cp.asnumpy(probs)  # copy to cpu anyway
            self.tran_prob_mutation = cp.asnumpy(self.tran_prob_mutation)
            self.tran_prob_mutation_free = cp.asnumpy(self.tran_prob_mutation_free)
        # nsite, ncell = len(reads), len(reads[0])
        tree = tree.copy()
        ncell, nsite, _ = probs.shape
        tran_prob_mut_broadcast = xp.tile(self.tran_prob_mutation, (nsite, 1)).reshape(
            nsite, probs.shape[-1], -1
        )
        tran_prob_mut_free_broadcast = xp.tile(
            self.tran_prob_mutation_free, (nsite, 1)
        ).reshape(nsite, probs.shape[-1], -1)
        log_likelihoods = []
        trees = []
        for node1 in self.traversor(tree):
            t = tree.copy()
            t1 = tree.copy()
            if node1.is_root():
                for node2 in self.traversor(t):
                    if node2.is_leaf():
                        node2.state = probs[int(node2.name)]
                        if reads is not None:
                            node2.reads = [r[int(node2.name)] for r in reads]
                        if masks is not None:
                            node2.masks = [m[int(node2.name)] for m in masks]
                        # print(reads)
                    else:
                        components = []
                        components_arg = {}
                        for child in node2.get_children():
                            state = child.state
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            val_, arg_ = util.log_mat_vec_max(
                                tran_prob_mut_free_broadcast, state
                            )
                            components.append(val_)
                            components_arg[child.identifier] = arg_
                        node2.state = xp.sum(xp.array(components), axis=0)
                        node2.arg = components_arg
                if return_trees:
                    for tnode in self.traversor(t1):
                        if tnode.is_root():
                            tnode.state = (
                                t.root.state.get() if use_gpu else t.root.state
                            )
                            tnode.arg = (
                                util.to_numpy(t.root.arg) if use_gpu else t.root.arg
                            )
                        elif not tnode.is_leaf():
                            tnode.arg = (
                                util.to_numpy(t[tnode.name].arg)
                                if use_gpu
                                else t[tnode.name].arg
                            )
                            # print(tnode.arg)
                    trees.append(t1)
                log_likelihoods.append(t.root.state[:, self.index_gt(0, 2)])
                log_likelihoods.append(t.root.state[:, self.index_gt(1, 1)])
                log_likelihoods.append(t.root.state[:, self.index_gt(2, 0)])
                continue
            for node2 in self.traversor(t):
                if node2.is_leaf():
                    node2.state = probs[int(node2.name)]
                    if reads is not None:
                        node2.reads = [r[int(node2.name)] for r in reads]
                    if masks is not None:
                        node2.masks = [m[int(node2.name)] for m in masks]
                    # print(reads)
                else:
                    if node1.parent == node2:
                        state = t[node1.identifier].state
                        state = state.reshape(state.shape[0], 1, state.shape[1])
                        sibs = node1.get_siblings()
                        # print(tran_prob_mut_broadcast.shape, state.shape)
                        val, arg = util.log_mat_vec_max(tran_prob_mut_broadcast, state)
                        components = [val]
                        components_arg = {node1.identifier: arg}
                        for sib in sibs:
                            state = t[sib.identifier].state
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            val_, arg_ = util.log_mat_vec_max(
                                tran_prob_mut_free_broadcast, state
                            )
                            components.append(val_)
                            components_arg[sib.identifier] = arg_
                    else:
                        components = []
                        components_arg = {}
                        for child in node2.get_children():
                            state = child.state
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            val_, arg_ = util.log_mat_vec_max(
                                tran_prob_mut_free_broadcast, state
                            )
                            components.append(val_)
                            components_arg[child.identifier] = arg_.astype(xp.int8)
                    node2.state = xp.sum(xp.array(components), axis=0)
                    node2.arg = components_arg
            log_likelihood = t.root.state[:, self.index_gt(2, 0)]
            log_likelihoods.append(log_likelihood)
            if return_trees:
                for tnode in self.traversor(t1):
                    if tnode.is_root():
                        tnode.state = t.root.state.get() if use_gpu else t.root.state
                        tnode.arg = util.to_numpy(t.root.arg) if use_gpu else t.root.arg
                    elif not tnode.is_leaf():
                        tnode.arg = (
                            util.to_numpy(t[tnode.name].arg)
                            if use_gpu
                            else t[tnode.name].arg
                        )
                trees.append(t1)
            # break
        log_likelihoods = xp.array(log_likelihoods)
        max_L = log_likelihoods.max(axis=0).sum()
        # break
        if return_trees:
            return max_L, log_likelihoods, trees
        return max_L

    def _bfs(self, node, site_index, state_index):
        cn_profile = self.cn_profile_at_index(state_index)
        node.cn = cn_profile
        if not node.is_leaf():
            for child in node.get_children():
                child_state_index = node.arg[child.identifier][site_index][state_index]
                self._bfs(child, site_index, int(child_state_index))

    def viterbi_decoding(self, probs, tree, sites, use_gpu=True):
        # total 2n+1 trees
        num_cell, num_site, _ = probs.shape
        decoded_trees = []
        max_L, likelihoods, trees = self.maximal_evaluate(
            probs, tree, return_trees=True, use_gpu=use_gpu
        )
        for site in sites:
            L = likelihoods[:, site]
            arg_max = int(L.argmax())
            gt = (2, 0)
            if arg_max == 2 * num_cell - 2:
                gt = (0, 2)
                arg_max = -1
            if arg_max == 2 * num_cell - 1:
                gt = (1, 1)
                arg_max = -1
            if arg_max == 2 * num_cell:
                arg_max = -1
            max_tree = trees[arg_max].copy()
            self._bfs(max_tree.root, site, self.index_gt(gt[0], gt[1]))
            decoded_trees.append(max_tree)
        return decoded_trees

    def genotype_calling(self, probs, tree):
        num_cell, num_site, _ = probs.shape
        decoded_trees = self.viterbi_decoding(probs, tree, range(num_site))
        genotypes = np.zeros((num_site, num_cell), dtype=int)
        for i in range(num_site):
            max_tree = decoded_trees[i]
            for leaf in max_tree.get_leaves():
                leaf = max_tree[leaf]
                cn_profile = leaf.cn
                if cn_profile[1] > 0:
                    genotypes[i, int(leaf.name)] = 1
        return genotypes


def construct_genotype(tree, indices):
    # get node list
    node_lists = []
    traversor = util.TraversalGenerator()
    for node in traversor(tree):
        if not node.is_root():
            node_lists.append(node)
        else:
            node_lists += [tree.root, tree.root, tree.root]
    # print(node_lists)
    nsite = len(indices)
    ncell = len(tree.get_leaves())
    genotypes = np.zeros((nsite, ncell), dtype=int)
    for i, ind in enumerate(indices):
        if ind != 2 * ncell:
            node = node_lists[ind.tolist()]
            idx = [int(leaf.name) for leaf in node.get_leaves()]
            genotypes[i, idx] = 1
    return genotypes


def estimate_copy_number(copies, tree):
    nums = []
    for copy in copies:
        estimator = CNEstimator(copy)
        nums.append(estimator(tree))
    return np.mean(nums)


def find_copy_gain_loss_on_branch(decoded_trees, gene_names=None, allele=1, loh=True):
    if not gene_names:
        gene_names = [f"gene_{i}" for i in range(len(decoded_trees))]
    traversor = util.TraversalGenerator()
    tree = decoded_trees[0].copy()  # a fresh tree
    for node in traversor(tree):
        node.events = {"loss": [], "gain": []}
    for d_tree, gene_name in zip(decoded_trees, gene_names):
        for node in traversor(d_tree):
            if node.is_root():
                if node.cn[allele] != 0:
                    tree[node.name].events["gain"].append(gene_name)
            else:
                if node.cn[allele] > node.parent.cn[allele]:
                    tree[node.name].events["gain"].append(gene_name)
                if node.cn[allele] < node.parent.cn[allele] and (
                    node.cn[allele] == 0 if loh else True
                ):
                    tree[node.name].events["loss"].append(gene_name)
    return tree


def map_copy_gain_and_loss(
    reads,
    tree,
    loci=None,
    cell_names=None,
    site_names=None,
    cn_min=1,
    cn_max=5,
    ado=0.1,
    seq_error=0.01,
    af=0.5,
    cn_noise=0.05,
    allele=1,  # 0: wildtype 1: mutant
    loh=True,  # loh deletion only
    use_gpu=False,
):
    if loci is None:
        loci = site_names
    assert len(loci) > 0, "loci is empty."
    assert cn_min > 0, "cn_min should be greater than 0."
    n_sites, n_cells, _ = reads.shape
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    if site_names is None:
        site_names = util.get_default_site_names(n_sites)
    sites = []
    for locus in loci:
        assert locus in site_names, f"{locus} is not in site_names."
        sites.append(site_names.index(locus))

    start_tree, _ = external.infer_scistree2_tree(reads, cell_names=cell_names)
    # need to convert back to numerical labels.
    start_tree = util.relabel(
        start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)}
    )
    cn_avg = estimate_copy_number(reads[:, :, -1], start_tree)

    s = ScisTreeCNA(
        CN_MAX=cn_max,
        CN_MIN=cn_min,
        LAMBDA_C=cn_avg,
        LAMBDA_S=1,
        LAMBDA_T=2 * n_cells - 1,
        verbose=False,
    )
    probs = s.init_prob_leaves_gpu(
        reads, ado=ado, seqerr=seq_error, cnerr=cn_noise, af=af
    )
    trees = s.viterbi_decoding(probs, tree, sites, use_gpu=use_gpu)
    mapped_tree = find_copy_gain_loss_on_branch(
        trees, gene_names=loci, allele=allele, loh=loh
    )
    return mapped_tree


def infer(
    reads,
    cell_names=None,
    cn_min=1,
    cn_max=5,
    ado=0.1,
    seq_error=0.01,
    af=0.5,
    cn_noise=0.05,
    max_iter=0,
    tree_batch_size=64,
    node_batch_size=64,
    true_tree=None,
    verbose=True,
    start_tree=None,
    verbose_mode="all",
):
    assert cn_min > 0, "cn_min should be greater than 0."
    n_sites, n_cells, _ = reads.shape
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)

    if start_tree is None:
        start_tree, _ = external.infer_scistree2_tree(reads, cell_names=cell_names)
        # need to convert back to numerical labels.
        start_tree = util.relabel(
            start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)}
        )
    else:
        if isinstance(start_tree, str):
            start_tree = util.from_newick(start_tree)
        elif isinstance(start_tree, util.BaseTree):
            start_tree = start_tree
        else:
            raise Exception("start tree is invalid.")
        
    if true_tree is not None and isinstance(true_tree, util.BaseTree):
        true_tree = util.relabel(
            true_tree, name_map={name: str(i) for i, name in enumerate(cell_names)}
        )
    cn_avg = estimate_copy_number(reads[:, :, -1], start_tree)
    max_iter = max_iter if max_iter > 0 else np.inf
    if verbose:
        console.rule("[bold red]ScisTreeCNA")
        console.print(f"#Cell: {n_cells} #Site: {n_sites}", justify="center")
        console.print(
            f"CN_MIN: {cn_min} CN_MAX: {cn_max} ADO: {ado} SEQ_ERR: {seq_error} CN_NOISE: {cn_noise}",
            justify="center",
        )
        console.print(
            f"MAX_ITER: {max_iter} TREE_BATCH_SIZE: {tree_batch_size} NODE_BATCH_SIZE: {node_batch_size}",
            justify="center",
        )
        console.rule("[bold red]Local Search")

    s = ScisTreeCNA(
        CN_MAX=cn_max,
        CN_MIN=cn_min,
        LAMBDA_C=cn_avg,
        LAMBDA_S=1,
        LAMBDA_T=2 * n_cells - 1,
        verbose=False,
    )
    probs = s.init_prob_leaves_gpu(
        reads, ado=ado, seqerr=seq_error, cnerr=cn_noise, af=af
    )
    tree, likelihood = s.local_search_batch(
        probs,
        start_tree,
        max_iter=max_iter,
        tree_batch_size=tree_batch_size,
        node_batch_size=node_batch_size,
        ground_truth=true_tree,
        verbose=verbose,
        verbose_mode=verbose_mode,
    )
    ml2, indices = s.marginal_evaluate_dp(probs, tree)
    geno = construct_genotype(tree, indices)
    tree = util.relabel(
        tree, name_map={str(i): name for i, name in enumerate(cell_names)}
    )

    console.rule()
    return tree, geno


def evaluate(
    reads,
    tree,
    cell_names=None,
    cn_min=1,
    cn_max=5,
    ado=0.1,
    seq_error=0.01,
    af=0.5,
    cn_noise=0.05,
):
    assert cn_min > 0, "cn_min should be greater than 0."
    n_sites, n_cells, _ = reads.shape
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    start_tree, _ = external.infer_scistree2_tree(reads, cell_names=cell_names)
    # need to convert back to numerical labels.
    start_tree = util.relabel(
        start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)}
    )
    cn_avg = estimate_copy_number(reads[:, :, -1], start_tree)

    s = ScisTreeCNA(
        CN_MAX=cn_max,
        CN_MIN=cn_min,
        LAMBDA_C=cn_avg,
        LAMBDA_S=1,
        LAMBDA_T=2 * n_cells - 1,
        verbose=False,
    )
    probs = s.init_prob_leaves_gpu(
        reads, ado=ado, seqerr=seq_error, cnerr=cn_noise, af=af
    )
    ml, indices = s.marginal_evaluate_dp(probs, tree)
    geno = construct_genotype(tree, indices)
    tree = util.relabel(
        tree, name_map={str(i): name for i, name in enumerate(cell_names)}
    )
    return ml, geno


def bootstrapping(
    reads,
    tree,
    n_bootstrap=10,
    n_site=-1,
    cell_names=None,
    cn_min=1,
    cn_max=5,
    ado=0.1,
    seq_error=0.01,
    af=0.5,
    cn_noise=0.05,
    max_iter=0,
    tree_batch_size=64,
    node_batch_size=64,
    true_tree=None,
    verbose=True,
    verbose_mode="all",
):
    pass