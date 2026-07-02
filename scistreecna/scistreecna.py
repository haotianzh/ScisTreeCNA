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


def _allele_specific_cn_logterm_gpu(cn_maj, cn_min, g0_arr, g1_arr, cn_err):
    """Log copy-number emission term for allele-specific input.

    Computes, for every (site, cell) and every genotype index, the log of

        P(cn_maj, cn_min | g0, g1)
            = sum_{k=0}^{g0} Binom(k; g0, 1/2) * Phi({cn_maj,cn_min} | {g1+k, g0-k})

    where k = number of wild-type copies sitting on the mutated homolog, so the
    true (unordered) allele-specific copy pair is {g1+k, g0-k}, and

        Phi({M,m} | {c1,c2}) = Pcn(M|c1)Pcn(m|c2) + Pcn(M|c2)Pcn(m|c1)   (single term if c1==c2)
        Pcn(o|t)             = (1-cn_err)*1[o==t] + cn_err*Poisson(o; lambda=t)

    The symmetric Phi marginalizes over which homolog the caller labelled
    "major" (so no phasing is needed). Poisson direction (observed ~ Poisson(true))
    matches the total-CN kernel. Missing observations (cn_maj<0 or cn_min<0) yield 0
    (the copy term drops out, leaving the genotype free, as for missing total CN).

    Args:
        cn_maj, cn_min: cupy float arrays (nsite, ncell). -1 marks a missing value.
        g0_arr, g1_arr: cupy int arrays (N,) giving (g0, g1) for each genotype index.
        cn_err:         copy-number error rate xi (float).

    Returns:
        cupy float32 array (nsite, ncell, N). Math is done in float64.
    """
    NEG_INF = -cp.inf
    xi = float(cn_err)
    log_xi = float(np.log(xi)) if xi > 0 else float("-inf")
    log_1mxi = float(np.log1p(-xi)) if xi < 1 else float("-inf")
    nsite, ncell = cn_maj.shape
    N = int(g0_arr.shape[0])

    # log-factorial lookup table, sized to cover all observed / true CN values.
    obs_max = 0
    for arr in (cn_maj, cn_min):
        a = cp.where(arr < 0, 0, arr)
        obs_max = max(obs_max, int(a.max().item()) if a.size else 0)
    maxidx = max(obs_max, int(g0_arr.max().item()) + int(g1_arr.max().item())) + 2
    logfact = cp.concatenate(
        [cp.zeros(1, cp.float64), cp.cumsum(cp.log(cp.arange(1, maxidx + 1, dtype=cp.float64)))]
    )

    M = cn_maj.astype(cp.float64)[:, :, None]   # (nsite, ncell, 1)
    mn = cn_min.astype(cp.float64)[:, :, None]
    Mi = cp.clip(M, 0, maxidx).astype(cp.int64)
    mni = cp.clip(mn, 0, maxidx).astype(cp.int64)
    g0 = g0_arr.astype(cp.float64)[None, None, :]   # (1, 1, N)
    g1 = g1_arr.astype(cp.float64)[None, None, :]
    g0i = g0_arr.astype(cp.int64)[None, None, :]

    def logpois(o_f, o_i, t_f):
        # log Poisson(o; lambda=t), with the t==0 branch (mass only at o==0) handled exactly.
        t_safe = cp.where(t_f == 0, 1.0, t_f)
        lp = o_f * cp.log(t_safe) - t_f - logfact[o_i]
        return cp.where(t_f == 0, cp.where(o_f == 0, 0.0, NEG_INF), lp)

    def logpcn(o_f, o_i, t_f):
        a = log_xi + logpois(o_f, o_i, t_f)
        deg = cp.where(o_f == t_f, log_1mxi, NEG_INF)
        return cp.logaddexp(a, deg)

    acc = cp.full((nsite, ncell, N), NEG_INF, dtype=cp.float64)
    kmax = int(g0_arr.max().item())
    log2 = float(np.log(2.0))
    for k in range(kmax + 1):
        c1 = g1 + k                              # mutated-homolog true CN
        c2 = g0 - k                              # wild-homolog true CN
        g0mk_i = cp.clip(g0i - k, 0, maxidx)
        logbin = logfact[g0i] - logfact[k] - logfact[g0mk_i] - g0 * log2   # (1,1,N)
        a = logpcn(M, Mi, c1) + logpcn(mn, mni, c2)
        b = logpcn(M, Mi, c2) + logpcn(mn, mni, c1)
        logphi = cp.where(c1 == c2, a, cp.logaddexp(a, b))
        term = logbin + logphi
        term = cp.where(g0i >= k, term, NEG_INF)   # mask invalid k (k > g0)
        acc = cp.logaddexp(acc, term)

    missing = (cn_maj < 0) | (cn_min < 0)
    acc = cp.where(missing[:, :, None], 0.0, acc)
    return acc.astype(cp.float32)


class NodeBatchLoader:
    """Yields nodes across several trees in node-count-bounded batches.

    Used by node-parallel passes: either flat ("all") or grouped into
    topological layers ("up"/"down") so every node in a yielded batch can be
    processed independently on the GPU. batch_size counts nodes, not trees.
    """

    def __init__(self, trees, batch_size):
        self.trees = trees
        self.batch_size = batch_size  # number of trees
        self.traversor = util.TraversalGenerator()
        self.orders = ["up", "down", "all"]

    def get_all_nodes(self):
        """Flatten all nodes of all trees, tagging each with its tree id (tid)."""
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
        """Yield node batches. order="all" flattens every node; "up"/"down"
        yield one set of batches per topological layer so a batch never mixes a
        parent with its child."""
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
    """Yields lists of candidate trees in batches of batch_size for whole-tree
    batch evaluation (marginal_evaluate_dp_batch)."""

    def __init__(self, trees, batch_size=128):
        self.trees = trees
        self.batch_size = batch_size

    def __len__(self):
        return len(self.trees)

    def __call__(self):
        """Yield successive slices of self.trees of at most batch_size trees."""
        num_batch = (len(self) + self.batch_size - 1) // self.batch_size
        for b in range(num_batch):
            yield self.trees[b * self.batch_size : (b + 1) * self.batch_size]


class ScisTreeCNA:
    """GPU JSC inference engine: holds the genotype state space, the two
    log-space branch transition matrices, and all likelihood / search / decoding
    routines.

    Args:
        CN_MAX, CN_MIN: copy-number bounds; together they fix the set of valid
            (g0, g1) genotype states and N = number of states.
        LAMBDA_C, LAMBDA_S, LAMBDA_T: rates for the transition solver
            (copy-number mutation, point mutation, and timeout/branch-length).
        verbose: if True, print the solved transition matrices.
    """

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
        """True iff (g0, g1) is a non-negative genotype with total CN in [CN_MIN, CN_MAX]."""
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def _indexing(self):
        """Build the (g0, g1) <-> flat-state-index maps (state2index / index2state)
        via triangular indexing, offset so the first valid total CN maps to 0."""
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
        """Flat state index for genotype (g0=i, g1=j)."""
        return self.state2index[(i, j)]

    def cn_profile_at_index(self, index):
        """Inverse of index_gt: return the (g0, g1) genotype for a flat state index."""
        return self.index2state[index]

    def index(self, i, j, p, q):
        """Flat index into the NxN transition matrix for the (i,j)->(p,q) entry."""
        # use yang's tri indexing
        i1 = self.index_gt(i, j)
        i2 = self.index_gt(p, q)
        return int(i1 * self.N + i2)

    def preprocess_reads_with_missing_values(self, reads):
        """Return reads unchanged plus a (nsite, ncell) bool mask of missing
        entries, identified by a total-CN value of -1."""
        cn = reads[:, :, 2]
        mask = cn == -1  # -1 indicates a missing
        # reads[:, :, 2][mask] = -1
        return reads, mask

    def _allele_specific_cn_logterm(self, cn_maj, cn_min, cn_err):
        """Build (g0, g1) index arrays and dispatch to the vectorized allele-specific term."""
        g0_np = np.zeros(self.N, dtype=np.int64)
        g1_np = np.zeros(self.N, dtype=np.int64)
        for idx, (i, j) in self.index2state.items():
            g0_np[idx] = i
            g1_np[idx] = j
        return _allele_specific_cn_logterm_gpu(
            cn_maj, cn_min, cp.asarray(g0_np), cp.asarray(g1_np), cn_err
        )

    def init_prob_leaves_gpu(self, reads, ado=0.1, seqerr=0.001, cnerr=0.2, af=None):
        # Total-CN input: reads[..., :3] = (ref, alt, total_cn).
        # Allele-specific input: reads[..., :4] = (ref, alt, cn_major, cn_minor).
        reads = cp.asarray(reads, dtype=np.float32)
        reads, mask = self.preprocess_reads_with_missing_values(reads)
        nsite, ncell = reads.shape[:2]
        allele_specific = reads.shape[2] >= 4
        ref = reads[:, :, 0]
        alt = reads[:, :, 1]
        if allele_specific:
            cn_maj = reads[:, :, 2]
            cn_min = reads[:, :, 3]
            # total copy number, missing iff either allele is missing
            cn = cp.where((cn_maj < 0) | (cn_min < 0), cp.float32(-1), cn_maj + cn_min)
        else:
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
        # In allele-specific mode, feed cn=-1 so the kernel returns only the reads
        # term (the allele-specific copy term is added separately, below). This reuses
        # the exact reads likelihood used by the total-CN path.
        cn_for_kernel = (
            cp.full((nsite, ncell), -1.0, dtype=cp.float32) if allele_specific else cn
        )
        util.compute_genotype_log_probs_cn_noise()(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                ref.ravel(),
                alt.ravel(),
                cn_for_kernel.ravel(),
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
        probs = probs.reshape((nsite, ncell, N))
        if allele_specific:
            probs = probs + self._allele_specific_cn_logterm(cn_maj, cn_min, cnerr)
        probs = cp.ascontiguousarray(cp.transpose(probs, (1, 0, 2)))
        # mask = mask.T # (nsite, ncell) -> (ncell, nsite)
        return probs

    def pairwise_distance_matrix(self, probs):
        """Expected genotype-mismatch distance between every pair of cells.

        probs is log P(genotype) per (cell, site, state); exponentiating gives
        per-state probabilities and the einsum sums, over sites and states, the
        probability that cell i and cell j disagree, yielding an (ncell, ncell)
        distance matrix for neighbor joining."""
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
        """Compute block/grid sizes for a batched (h, w) elementwise/matmul kernel.

        x indexes rows (h, the sites), y indexes columns (w, the states), and the
        grid's z-dimension is the batch (one tree-node slice per z)."""
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
        """Batched log-matmul dst = src @ mat2 over nb node slices.

        ptr_src/ptr_dst are GPU int arrays of base+index*stride byte pointers, one
        per slice; the loop chunks the batch so the grid z-dim never exceeds the
        CUDA limit (MAX_GRID_Z)."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_size, grid_size = self._get_block_grid(h, w, chunk)
            kernel(grid_size, block_size,
                   (ptr_src[off:off+chunk], mat2, ptr_dst[off:off+chunk], chunk, h, w),
                   shared_mem=shared_mem)

    def _launch_batched_add_matmul(self, kernel, ptr_a, ptr_b, mat2, ptr_dst, nb, h, w, shared_mem):
        """Batched fused dst = (a + b) @ mat2 in log space, chunked over MAX_GRID_Z.
        Pointer args are per-slice base+index*stride byte-pointer arrays."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_size, grid_size = self._get_block_grid(h, w, chunk)
            kernel(grid_size, block_size,
                   (ptr_a[off:off+chunk], ptr_b[off:off+chunk], mat2, ptr_dst[off:off+chunk],
                    chunk, h, w),
                   shared_mem=shared_mem)

    def _launch_batched_matadd(self, kernel, ptr_a, ptr_b, ptr_dst, nb, h, w):
        """Batched elementwise log-add dst = a + b (logaddexp of two child U's),
        chunked over MAX_GRID_Z. Pointer args are per-slice byte-pointer arrays."""
        for off in range(0, nb, self._MAX_GRID_Z):
            chunk = min(self._MAX_GRID_Z, nb - off)
            block_size, grid_size = self._get_block_grid(h, w, chunk)
            kernel(grid_size, block_size,
                   (ptr_a[off:off+chunk], ptr_b[off:off+chunk], ptr_dst[off:off+chunk],
                    chunk, h, w))

    def _launch_batched_3vecdot(self, kernel, ptr_a, ptr_b, ptr_c, out, nb, h, w):
        """Batched per-site three-vector log dot-product (a+b+c, then log-sum over
        states) -> out (nb, h); used by PM-placement scoring. Chunked over MAX_GRID_Z."""
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
        def _sibling(n):
            """Fast sibling lookup for binary trees (avoids get_siblings() list creation)."""
            pc = n.parent._children
            return pc[1] if pc[0] is n else pc[0]

        node_id_map = {}  # id(node) -> flat index
        nr_self_l = []
        nr_sib_l = []
        nr_par_l = []
        root_list = []
        idx = 0
        for tid, tree in enumerate(trees):
            nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
            for nid, node in nodes_dict.items():
                node_id_map[id(node)] = idx
                node.tid = tid
                idx += 1
        total_nodes = idx
        _get = node_id_map.__getitem__

        # Build scoring indices in single pass (merged with id_map)
        for tid, tree in enumerate(trees):
            nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
            for nid, node in nodes_dict.items():
                nidx = _get(id(node))
                if node.is_root():
                    root_list.append(nidx)
                else:
                    nr_self_l.append(nidx)
                    nr_sib_l.append(_get(id(_sibling(node))))
                    nr_par_l.append(_get(id(node.parent)))

        # Build topological layers with pre-computed index arrays (numpy first, then GPU)
        _np_int64 = np.int64
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
                    cp.asarray(np.array(leaf_idx, dtype=_np_int64)),
                    cp.asarray(np.array(leaf_cells, dtype=_np_int64))))
            if int_idx:
                up_layers.append(('internal',
                    cp.asarray(np.array(int_idx, dtype=_np_int64)),
                    cp.asarray(np.array(int_c0, dtype=_np_int64)),
                    cp.asarray(np.array(int_c1, dtype=_np_int64))))

        # Derive down-pass layers by reversing raw up-pass layers (avoids second topo sort)
        # Skip leaf nodes: Q[leaf] is never read (leaves have no children, scoring uses Q[parent])
        down_layers = []
        for layer in reversed(layers_up):
            nr_idx = []
            nr_par = []
            nr_sib = []
            has_root = False
            for n in layer:
                if n.is_root():
                    has_root = True
                elif not n.is_leaf():  # skip leaves — their Q is never used
                    nr_idx.append(_get(id(n)))
                    nr_par.append(_get(id(n.parent)))
                    nr_sib.append(_get(id(_sibling(n))))
            if has_root:
                down_layers.append(('root',))
            if nr_idx:
                down_layers.append(('internal',
                    cp.asarray(np.array(nr_idx, dtype=_np_int64)),
                    cp.asarray(np.array(nr_par, dtype=_np_int64)),
                    cp.asarray(np.array(nr_sib, dtype=_np_int64))))

        nr_self_gpu = cp.asarray(np.array(nr_self_l, dtype=_np_int64))
        nr_sib_gpu = cp.asarray(np.array(nr_sib_l, dtype=_np_int64))
        nr_par_gpu = cp.asarray(np.array(nr_par_l, dtype=_np_int64))
        root_gpu = cp.asarray(np.array(root_list, dtype=_np_int64))

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
        Bottom-up — optimized using contiguous arrays + batched kernels (same as batch version).
        Still stores results on nodes for compatibility with calculate_Q and marginal_evaluate_dp.

        Single-tree Felsenstein IN pass. For each node computes three (nsite, N)
        log-likelihood tables:
            U  = likelihood of the subtree given the node's genotype,
            U_ = U pushed up a mutation-free branch (tran_prob_mutation_free),
            _U = U pushed up the branch that carries the one point mutation
                 (tran_prob_mutation).
        Results are written back onto node.U / node.U_ / node._U. (misspelled
        name kept intentionally.)
        """
        h, w = probs.shape[1:]
        shared_mem = w * w * 4
        stride = h * w * 4

        # Build topology
        node_id_map = {}
        node_list = []
        idx = 0
        nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
        for nid, node in nodes_dict.items():
            node_id_map[id(node)] = idx
            node_list.append(node)
            idx += 1
        total_nodes = idx
        _get = node_id_map.__getitem__

        def _sibling(n):
            pc = n.parent._children
            return pc[1] if pc[0] is n else pc[0]

        # Topological sort (single tree)
        from .topological_sort import topological_sort
        layers_up = topological_sort(tree, order="up")

        # Contiguous arrays
        _buf = cp.zeros((3 * total_nodes, h, w), dtype=cp.float32)
        all_U  = _buf[0*total_nodes:1*total_nodes]
        all_U_ = _buf[1*total_nodes:2*total_nodes]
        all__U = _buf[2*total_nodes:3*total_nodes]
        base_U  = all_U.data.ptr
        base_U_ = all_U_.data.ptr
        base__U = all__U.data.ptr

        _matmul = util.batch_log_matmul_cuda()
        _add_matmul = util.batch_add_log_matmul_cuda()
        _matadd = util.batch_matadd_cuda()
        _np_int64 = np.int64

        for layer in layers_up:
            leaves = [n for n in layer if n.is_leaf()]
            internals = [n for n in layer if not n.is_leaf()]
            if leaves:
                idx_l = np.array([_get(id(n)) for n in leaves], dtype=_np_int64)
                cells = np.array([int(n.name) for n in leaves], dtype=_np_int64)
                indices = cp.asarray(idx_l)
                cell_ids = cp.asarray(cells)
                nb = len(indices)
                all_U[indices] = probs[cell_ids]  # leaf U = its cell's per-state log-emission
                ptr_u  = base_U  + indices * stride
                ptr_u_ = base_U_ + indices * stride
                ptr__u = base__U + indices * stride
                self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
                self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)
            if internals:
                idx_i = np.array([_get(id(n)) for n in internals], dtype=_np_int64)
                c0 = np.array([_get(id(n._children[0])) for n in internals], dtype=_np_int64)
                c1 = np.array([_get(id(n._children[1])) for n in internals], dtype=_np_int64)
                indices = cp.asarray(idx_i)
                c0_gpu = cp.asarray(c0)
                c1_gpu = cp.asarray(c1)
                nb = len(indices)
                ptr_c0_u_ = base_U_ + c0_gpu * stride
                ptr_c1_u_ = base_U_ + c1_gpu * stride
                ptr_u  = base_U  + indices * stride
                ptr_u_ = base_U_ + indices * stride
                ptr__u = base__U + indices * stride
                # U_ = (U_[c0] + U_[c1]) @ tran_mut_free; U = U_[c0] + U_[c1]; _U = U @ tran_mut
                self._launch_batched_add_matmul(_add_matmul, ptr_c0_u_, ptr_c1_u_, self.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
                self._launch_batched_matadd(_matadd, ptr_c0_u_, ptr_c1_u_, ptr_u, nb, h, w)
                self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)

        # Copy results back to node attributes for compatibility
        for node in node_list:
            nidx = _get(id(node))
            node.U  = all_U[nidx]
            node.U_ = all_U_[nidx]
            node._U = all__U[nidx]
        return 0

    def calculate_Q(self, tree):
        """
        calcuate Q for each site recursively, have to do after calculation of U, use this fashion instead of DFS

        Top-down Felsenstein OUT pass. node.Q[site] is the (N, N) log table giving,
        for each genotype at the node, the likelihood of everything outside its
        subtree. Computed pre-order: root.Q is the identity; a child's Q starts
        from its parent's Q, adds each sibling's mutation-free up-message (sib.U_),
        then propagates across the node's own mutation-free branch.
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

        Single-tree marginal log-likelihood with the one point mutation placed on
        its best branch, per site. Runs the U (IN) and Q (OUT) passes, then for
        every candidate branch combines _U (mutation message) with the siblings'
        mutation-free messages and the parent's outside table Q to score "PM on
        this branch"; the root branches score "no PM" / PM at the root.

        Returns:
            max_L:    sum over sites of the best per-site likelihood (best branch).
            indicies: argmax branch index per site (which branch carries the PM).
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
                # PM at the root: read U directly at the three founder genotypes
                likelihoods.append(node.U[:, self.index_gt(1, 1)])
                likelihoods.append(node.U[:, self.index_gt(0, 2)])
                likelihoods.append(node.U[:, self.index_gt(2, 0)])
                # print(node.name, likelihood)
                continue
            # PM on this branch: this node's mutation message + siblings' mutation-free
            # messages, propagated through the parent's outside table Q.
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
        indicies = cp.argmax(likelihoods, axis=0)  # best PM branch per site
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

        Scaffold/debug walk over internal nodes whose two children are both
        internal (quartet candidates); it identifies the four grandchildren but
        performs no swap. See nni_search_sinlge_round_batch for the working
        neighbor enumeration.
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

        One round of NNI: enumerate all quartet swaps (at internal nodes with two
        internal children, 2 rearrangements each) and triplet swaps (at every
        non-root internal node, 2 each), then batch-evaluate every neighbor plus
        the base tree. Returns the best (tree, likelihood) found this round.
        """
        candidates = []
        # Pre-collect node identifiers for quartet and triplet swaps
        nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
        quartet_nodes = []  # nodes where both children are internal
        triplet_nodes = []  # non-root, non-leaf nodes
        for nid, nd in nodes_dict.items():
            if nd.is_leaf():
                continue
            ch = nd._children
            if len(ch) == 2 and not ch[0].is_leaf() and not ch[1].is_leaf():
                quartet_nodes.append(nid)
            if not nd.is_root():
                triplet_nodes.append(nid)

        # Quartet swaps: at node nid with children p1,p2, exchange one grandchild
        # from p1's pair with one from p2's pair (two distinct rearrangements).
        _copy = struct_copy_tree
        for nid in quartet_nodes:
            t1 = _copy(tree)
            p1, p2 = t1[nid]._children
            lc1, lc2 = p1._children
            rc1, rc2 = p2._children
            p1.remove_child(lc1)
            p2.remove_child(rc1)
            lc1.set_parent(p2)
            rc1.set_parent(p1)
            p1.add_child(rc1)
            p2.add_child(lc1)
            candidates.append(t1)
            t2 = _copy(tree)
            p1, p2 = t2[nid]._children
            lc1, lc2 = p1._children
            rc1, rc2 = p2._children
            p1.remove_child(lc1)
            p2.remove_child(rc2)
            lc1.set_parent(p2)
            rc2.set_parent(p1)
            p1.add_child(rc2)
            p2.add_child(lc1)
            candidates.append(t2)

        # Triplet swaps: at node nd, exchange one of nd's children with nd's sibling
        # (two rearrangements, one per child of nd).
        for nid in triplet_nodes:
            t1 = _copy(tree)
            nd = t1[nid]
            par = nd.parent
            pc = par._children
            sib = pc[1] if pc[0] is nd else pc[0]
            c1, c2 = nd._children
            nd.remove_child(c1)
            par.remove_child(sib)
            nd.add_child(sib)
            sib.set_parent(nd)
            par.add_child(c1)
            c1.set_parent(par)
            candidates.append(t1)
            t2 = _copy(tree)
            nd = t2[nid]
            par = nd.parent
            pc = par._children
            sib = pc[1] if pc[0] is nd else pc[0]
            c1, c2 = nd._children
            nd.remove_child(c2)
            par.remove_child(sib)
            nd.add_child(sib)
            sib.set_parent(nd)
            par.add_child(c2)
            c2.set_parent(par)
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
                # map the within-batch argmax back to an index in all_trees
                best_tree = all_trees[int(bi * tree_batch_size + max_idx)]
        return best_tree, best_likelihood

    def nni_search_non_optim_sinlge_round(self, probs, tree):
        """
        NNI neighbor

        Reference (un-optimized) single NNI round: same quartet + triplet
        neighbor set as nni_search_sinlge_round_batch, but each candidate is
        evaluated one at a time via marginal_evaluate_dp. Returns the best
        (tree, likelihood).
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
        """Hill-climbing NNI local search (batched evaluation).

        Repeatedly applies nni_search_sinlge_round_batch, moving to the best
        neighbor each round, until the likelihood stops improving or max_iter is
        reached (max_iter=0 means unbounded). If ground_truth is given, logs tree
        accuracy / nRF each step. Returns the final (tree, likelihood).
        """
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
        """Reference (un-batched) NNI local search using
        nni_search_non_optim_sinlge_round; iterates until no improvement.
        Returns the final (tree, likelihood)."""
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
        """Max-product (Viterbi) pass: for each possible PM branch placement,
        run a max-sum upward pass storing per-node argmax backpointers (node.arg),
        and collect the per-site MAP log-likelihood at the root.

        There are 2n-1 internal/leaf branch placements (plus 3 root founder
        genotypes); iterating node1 over all branches selects which branch carries
        the mutation (via tran_prob_mutation) while other branches use the
        mutation-free matrix.

        Args:
            return_trees: also return the per-placement decoded trees (with .arg
                backpointers) for downstream Viterbi decoding.
            reads, masks: optional per-cell raw reads / missing masks to attach to
                leaves for reporting.
            use_gpu: run on CuPy (True) or NumPy (False).

        Returns:
            max_L, or (max_L, log_likelihoods, trees) when return_trees is True.
        """
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
        """Top-down traceback: set node.cn to the (g0, g1) for the chosen state,
        then recurse into children using the stored argmax backpointers
        (node.arg) to pick each child's MAP state for this site."""
        cn_profile = self.cn_profile_at_index(state_index)
        node.cn = cn_profile
        if not node.is_leaf():
            for child in node.get_children():
                child_state_index = node.arg[child.identifier][site_index][state_index]
                self._bfs(child, site_index, int(child_state_index))

    def viterbi_decoding(self, probs, tree, sites, use_gpu=True):
        """For each requested site, pick the MAP PM placement and trace back the
        per-node genotypes, returning one fully-decoded tree per site (each node
        annotated with .cn). The last three likelihood rows are the root founder
        genotypes (0,2)/(1,1)/(2,0); those map to a 'PM at root' decoding."""
        # total 2n+1 trees
        num_cell, num_site, _ = probs.shape
        decoded_trees = []
        max_L, likelihoods, trees = self.maximal_evaluate(
            probs, tree, return_trees=True, use_gpu=use_gpu
        )
        for site in sites:
            L = likelihoods[:, site]
            arg_max = int(L.argmax())  # best PM placement for this site
            gt = (2, 0)
            # the final three rows encode root founder genotypes (no internal-branch PM)
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
        """Call a binary (site x cell) mutation genotype matrix: a cell is mutant
        (1) at a site iff its decoded leaf genotype has g1 > 0 (mutant copies present)."""
        num_cell, num_site, _ = probs.shape
        decoded_trees = self.viterbi_decoding(probs, tree, range(num_site))
        genotypes = np.zeros((num_site, num_cell), dtype=int)
        for i in range(num_site):
            max_tree = decoded_trees[i]
            for leaf in max_tree.get_leaves():
                leaf = max_tree[leaf]
                cn_profile = leaf.cn
                if cn_profile[1] > 0:  # g1 (mutant copies) > 0 -> mutant
                    genotypes[i, int(leaf.name)] = 1
        return genotypes


def construct_genotype(tree, indices):
    """Turn per-site best-PM-branch indices (from marginal_evaluate_dp) into a
    binary (nsite, ncell) genotype matrix: cells under the PM branch are mutant.

    The branch list mirrors marginal_evaluate_dp's likelihood ordering, with the
    root entered three times (its three founder genotypes). The sentinel index
    2*ncell means "PM at the wild-type root" -> no cell is mutant for that site."""
    # get node list
    node_lists = []
    traversor = util.TraversalGenerator()
    for node in traversor(tree):
        if not node.is_root():
            node_lists.append(node)
        else:
            node_lists += [tree.root, tree.root, tree.root]  # 3 root founder-genotype slots
    # print(node_lists)
    nsite = len(indices)
    ncell = len(tree.get_leaves())
    genotypes = np.zeros((nsite, ncell), dtype=int)
    for i, ind in enumerate(indices):
        if ind != 2 * ncell:  # 2*ncell = PM at wild-type root -> all cells reference
            node = node_lists[ind.tolist()]
            idx = [int(leaf.name) for leaf in node.get_leaves()]  # cells in PM subtree
            genotypes[i, idx] = 1
    return genotypes


def total_copy_number(reads):
    """Per-(site, cell) total copy number, for either input layout.

    Total-CN input (..., 3) -> reads[..., 2]; allele-specific input (..., 4) ->
    cn_major + cn_minor, marked missing (-1) iff either allele is missing.
    """
    reads = np.asarray(reads)
    if reads.shape[-1] >= 4:
        cmaj = reads[:, :, 2]
        cmin = reads[:, :, 3]
        return np.where((cmaj < 0) | (cmin < 0), -1, cmaj + cmin)
    return reads[:, :, 2]


def estimate_copy_number(copies, tree):
    """Mean inferred ancestral/average copy number over all sites, used to set
    LAMBDA_C. `copies` is the per-site (cell) total-CN matrix; each site is fit on
    the given tree by CNEstimator."""
    nums = []
    for copy in copies:
        estimator = CNEstimator(copy)
        nums.append(estimator(tree))
    return np.mean(nums)


def find_copy_gain_loss_on_branch(decoded_trees, gene_names=None, allele=1, loh=True):
    """Annotate copy-number gain/loss events on tree branches.

    Given one decoded tree per gene/locus (from viterbi_decoding, each node
    carrying a .cn = (g0, g1)), compare each node's copy number of `allele`
    (0=wild-type, 1=mutant) against its parent: an increase is a 'gain', a
    decrease a 'loss' (restricted to losses reaching 0 when loh=True). Returns one
    tree whose nodes hold node.events = {'gain': [...], 'loss': [...]} listing the
    contributing gene names."""
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
    """Map copy-number gain/loss events onto branches of a given tree for the
    chosen `loci`.

    Builds a ScisTreeCNA model from `reads` (LAMBDA_C set from estimated average
    CN), computes leaf emission probs, Viterbi-decodes the selected sites on
    `tree`, then calls find_copy_gain_loss_on_branch. `loci`/`site_names` select
    which sites to map. Returns the event-annotated tree."""
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
    cn_avg = estimate_copy_number(total_copy_number(reads), start_tree)

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
    """Main entry point: infer a cell lineage tree and genotype matrix from reads.

    Pipeline: build an initial tree (ScisTree2 if start_tree is None, else the
    provided newick/tree) -> estimate average CN to set LAMBDA_T = 2n-1 and
    LAMBDA_C -> compute leaf emission probs -> NNI local search (local_search_batch)
    -> place the PM per site (marginal_evaluate_dp) and call genotypes.

    Args:
        reads: (n_sites, n_cells, 3 or 4) read/CN array (see module docstring).
        cn_min/cn_max, ado, seq_error, af, cn_noise: model parameters.
        max_iter: NNI iteration cap (0 = unbounded).
        tree_batch_size/node_batch_size: GPU batching for evaluation.
        true_tree: optional ground truth for accuracy logging.
        start_tree: optional initial tree (BaseTree or newick str).

    Returns:
        (tree, geno): inferred tree (relabelled to cell_names) and binary
        (nsite, ncell) genotype matrix.
    """
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
    cn_avg = estimate_copy_number(total_copy_number(reads), start_tree)
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
    """Score a fixed `tree` (no search): build the model from `reads`, compute the
    marginal log-likelihood with best per-site PM placement, and call genotypes.
    Returns (marginal_log_likelihood, genotype_matrix)."""
    assert cn_min > 0, "cn_min should be greater than 0."
    n_sites, n_cells, _ = reads.shape
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    start_tree, _ = external.infer_scistree2_tree(reads, cell_names=cell_names)
    # need to convert back to numerical labels.
    start_tree = util.relabel(
        start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)}
    )
    cn_avg = estimate_copy_number(total_copy_number(reads), start_tree)

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
    """Not implemented — placeholder stub for bootstrap support over resampled sites."""
    pass