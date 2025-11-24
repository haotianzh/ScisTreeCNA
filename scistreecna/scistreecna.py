import warnings
from contextlib import nullcontext
import scistree2 as s2
from phytreeviz import TreeViz  # for tree visuaization
import numpy as np
import cupy as cp
from time import time
from rich.console import Console
from rich import print
from . import util, external
from .cn_estimate import *
from .topological_sort import *
from .transition_solver import *


console = Console(force_jupyter=False, log_path=False, width=96)
console.is_jupyter = False
warnings.filterwarnings("ignore")  # opts on log 0 is normal, a -inf is always expected.
cp.set_printoptions(suppress=True)


# os.environ['OPENBLAS_NUM_THREADS'] = '1'
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

    def calculate_U_batch(self, probs, trees, batch_size=512):
        """
        U_ and _U is not necessarily be stored. computing is not actually time-consuming. (memalloc is bottleneck)
        """
        h, w = probs.shape[1:]
        loader = NodeBatchLoader(trees, batch_size)
        zeros = cp.zeros([3 * len(loader), h, w], dtype=cp.float32)
        cur = 0
        for nodes in loader(order="up"):
            # print(len(nodes))
            INIT = False
            # zeros = cp.zeros([3*len(nodes), h, w], dtype=cp.float32)
            # cur = 0
            for node in nodes:
                # what if I store pointers only
                node.U = zeros[cur]
                node._U = zeros[cur + 1]
                node.U_ = zeros[cur + 2]
                cur += 3
                if node.is_leaf():
                    node.U = probs[int(node.name)]
                    INIT = True

            # ts += time() - t
            # print([n.U.data.ptr for n in nodes])
            # t = time()
            ptr_u = cp.array([n.U.data.ptr for n in nodes])
            ptr_u_ = cp.array([n.U_.data.ptr for n in nodes])
            ptr__u = cp.array([n._U.data.ptr for n in nodes])
            block_size = (32, 32)
            grid_size = (
                (h + block_size[0] - 1) // block_size[0],
                (w + block_size[1] - 1) // block_size[1],
                len(nodes),
            )
            # ts += time() - t
            # calculate U
            if not INIT:
                ptr_c0_u_ = cp.array([n.get_children()[0].U_.data.ptr for n in nodes])
                ptr_c1_u_ = cp.array([n.get_children()[1].U_.data.ptr for n in nodes])
                util.batch_matadd_cuda()(
                    grid_size,
                    block_size,
                    (ptr_c0_u_, ptr_c1_u_, ptr_u, len(nodes), h, w),
                )
            # calculate U_ and _U
            util.batch_log_matmul_cuda()(
                grid_size,
                block_size,
                (ptr_u, self.tran_prob_mutation_free, ptr_u_, len(nodes), h, w),
            )
            util.batch_log_matmul_cuda()(
                grid_size,
                block_size,
                (ptr_u, self.tran_prob_mutation, ptr__u, len(nodes), h, w),
            )

    def calculate_Q_batch(self, probs, trees, batch_size=512):
        """
        Actually, only (2,0) need to be considered, this leads to less memory consumption: (num_site, k, k) -> (num_site, k)
        """
        h, w = probs.shape[1:]
        loader = NodeBatchLoader(trees, batch_size)
        zeros = cp.zeros([2 * len(loader), h, w], dtype=cp.float32)
        tran_prob_mutation_free_t = cp.ascontiguousarray(self.tran_prob_mutation_free.T)
        zeros[:, :, self.index_gt(2, 0)] = 1
        zeros = cp.log(zeros)
        cur = 0
        for nodes in loader(order="down"):
            INIT = False
            for node in nodes:
                node.Q = zeros[cur]
                node.Q_ = zeros[cur + 1]
                cur += 2
                if node.is_root():
                    INIT = True

            block_size = (32, 32)
            grid_size = (
                (h + block_size[0] - 1) // block_size[0],
                (w + block_size[1] - 1) // block_size[1],
                len(nodes),
            )
            # calculate Q
            if not INIT:
                ptr_par_q = cp.array([n.parent.Q.data.ptr for n in nodes])
                ptr_q = cp.array([n.Q.data.ptr for n in nodes])
                ptr_q_ = cp.array([n.Q_.data.ptr for n in nodes])
                ptr_u = cp.array([n.get_siblings()[0].U_.data.ptr for n in nodes])
                util.batch_matadd_cuda()(
                    grid_size, block_size, (ptr_par_q, ptr_u, ptr_q_, len(nodes), h, w)
                )
                util.batch_log_matmul_cuda()(
                    grid_size,
                    block_size,
                    (ptr_q_, tran_prob_mutation_free_t, ptr_q, len(nodes), h, w),
                )

    def marginal_evaluate_dp_batch(self, probs, trees, batch_size=512):
        h, w = probs.shape[1:]
        self.calculate_U_batch(probs, trees, batch_size)
        self.calculate_Q_batch(probs, trees, batch_size)
        loader = NodeBatchLoader(trees, batch_size=batch_size)
        likelihoods = cp.log(
            cp.zeros((len(loader) - len(trees), h), dtype=cp.float32)
        )  # ((n-1)) * n_tree (no mut on root) n: #all nodes
        likelihoods_root = cp.zeros((3 * len(trees), h), dtype=cp.float32)
        tids = []
        idx = 0
        idx_root = 0
        for nodes in loader(order="all"):
            # now have both U and Q, get max of all nodes
            ptr_u_ = []
            ptr__u = []
            ptr_u = []
            ptr_q = []
            tid = []
            for node in nodes:
                if node.is_root():
                    likelihoods_root[idx_root] = node.U[:, self.index_gt(0, 2)]
                    likelihoods_root[idx_root + 1] = node.U[:, self.index_gt(1, 1)]
                    likelihoods_root[idx_root + 2] = node.U[:, self.index_gt(2, 0)]
                    idx_root += 3
                    pass
                else:
                    ptr__u.append(node._U.data.ptr)
                    ptr_u_.append(node.get_siblings()[0].U_.data.ptr)
                    ptr_q.append(node.parent.Q.data.ptr)
                    ptr_u.append(node.U.data.ptr)
                tid.append(node.tid)
            ptr_u_ = cp.array(ptr_u_)
            ptr__u = cp.array(ptr__u)
            ptr_q = cp.array(ptr_q)
            ptr_u = cp.array(ptr_u)  # U is no longer needed
            block_size = (256, 1)
            grid_size = (((h + block_size[0] - 1) // block_size[0]), 1, len(ptr_u))
            res = likelihoods[idx : idx + len(ptr_u)]
            # batch_log_vecdot_cuda()(grid_size, block_size, (ptr_u, ptr_q, res, len(ptr_u), h, w))
            util.batch_log_3vecdot_cuda()(
                grid_size, block_size, (ptr__u, ptr_u_, ptr_q, res, len(ptr_u), h, w)
            )
            tids += tid
            idx += len(ptr_u)

        likelihoods = likelihoods.reshape(len(trees), -1, h)
        likelihoods = cp.concatenate(
            [likelihoods, likelihoods_root.reshape(len(trees), -1, h)], axis=1
        )
        likelihoods = likelihoods.max(axis=1).sum(axis=-1, dtype=cp.float64)
        return likelihoods  # , indices # (n_t,)

    def calcualte_U(self, tree, probs):
        """
        Bottom-up
        """
        h, w = probs.shape[1:]
        # print('h', h, 'w', w)
        block_size = (32, 32)
        grid_size = (
            (h + block_size[0] - 1) // block_size[0],
            (w + block_size[1] - 1) // block_size[1],
        )
        # print(grid_size, block_size)
        ts = 0
        for node in self.traversor(tree):
            if node.is_leaf():
                node.U = probs[int(node.name)].astype(cp.float32)
                node.U_ = cp.zeros([h, w], dtype=cp.float32)
                node._U = cp.zeros([h, w], dtype=cp.float32)
                util.log_matmul_cuda()(
                    grid_size,
                    block_size,
                    (node.U, self.tran_prob_mutation_free, node.U_, h, w),
                )
                util.log_matmul_cuda()(
                    grid_size,
                    block_size,
                    (node.U, self.tran_prob_mutation, node._U, h, w),
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
                grid_size,
                block_size,
                (node.U, self.tran_prob_mutation_free, node.U_, h, w),
            )
            util.log_matmul_cuda()(
                grid_size, block_size, (node.U, self.tran_prob_mutation, node._U, h, w)
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
        best_likelihood, indicies = self.marginal_evaluate_dp(probs, best_tree.copy())
        loader = TreeBatchLoader(candidates, batch_size=tree_batch_size)
        # print('tree len', len(loader))
        num_tree_evaulated = 0
        for bi, trees in enumerate(loader()):
            # print(f'#batch: {bi}')
            likelihoods = self.marginal_evaluate_dp_batch(
                probs, [_.copy() for _ in trees], batch_size=node_batch_size
            )
            num_tree_evaulated += len(trees)
            # pbar.update(num_tree_evaulated)
            max_idx = cp.argmax(likelihoods)
            max_lh = likelihoods[max_idx]
            if max_lh > best_likelihood:
                best_likelihood = max_lh
                best_tree = candidates[int(bi * tree_batch_size + max_idx)]
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
        iters = 0
        context = (
            console.status("[bold green]NNI Searching") if verbose else nullcontext()
        )
        with context:
            while True:
                better_tree, likelihood = self.nni_search_sinlge_round_batch(
                    probs,
                    tree,
                    tree_batch_size=tree_batch_size,
                    node_batch_size=node_batch_size,
                )
                if likelihood <= L:
                    if verbose:
                        console.log(
                            f"[bold red]Local search complete. Best Likelihood: {L}"
                        )
                    break
                else:
                    L = likelihood
                    tree = better_tree
                    str_log = f"\r[Iteration {iters}]\tLikelihood: {L:.4f}"
                    if ground_truth is not None and isinstance(
                        ground_truth, util.BaseTree
                    ):
                        str_log += f"\tTree accuracy: {util.tree_accuracy(ground_truth, tree):.4f}"
                    if verbose:
                        if verbose_mode == "all":
                            console.log(str_log)
                        else:
                            context.update(
                                f"[bold green]NNI Searching[/bold green]\t{str_log}"
                            )
                    iters += 1
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
                if node.cn[allele] < node.parent.cn[allele] and (node.cn[allele] == 0 if loh else True):
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
    loh=True, # loh deletion only
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
    mapped_tree = find_copy_gain_loss_on_branch(trees, gene_names=loci, allele=allele)
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
    tree_batch_size=64,
    node_batch_size=64,
    true_tree=None,
    verbose=True,
    verbose_mode="all",
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
    if true_tree is not None and isinstance(true_tree, util.BaseTree):
        true_tree = util.relabel(
            true_tree, name_map={name: str(i) for i, name in enumerate(cell_names)}
        )
    cn_avg = estimate_copy_number(reads[:, :, -1], start_tree)
    if verbose:
        console.rule("[bold red]ScisTreeCNA")
        console.print(f"#Cell: {n_cells} #Site: {n_sites}", justify="center")
        console.print(
            f"CN_MIN: {cn_min} CN_MAX: {cn_max} ADO: {ado} SEQ_ERR: {seq_error} CN_NOISE: {cn_noise}",
            justify="center",
        )
        console.print(
            f"TREE_BATCH_SIZE: {tree_batch_size} NODE_BATCH_SIZE: {node_batch_size}",
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
