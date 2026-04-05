"""Analyze whether root-only IN score can predict the full marginal likelihood ranking.
If the top-1 tree by root score == top-1 by full score, two-phase screening is viable."""
import time
import numpy as np
import cupy as cp
import scistreecna as scna
from scistreecna.scistreecna import ScisTreeCNA, TreeBatchLoader
from scistreecna.scistreecna import estimate_copy_number
from scistreecna.base.tree import struct_copy_tree
from scistreecna import util, external
from scistreecna.topological_sort import batch_topological_sort

def compute_root_only_scores(s, probs, trees):
    """Compute approximate tree likelihoods using only root IN values (U-pass only, no Q-pass)."""
    h, w = probs.shape[1:]
    shared_mem = w * w * 4
    stride = h * w * 4

    node_id_map = {}
    idx = 0
    for tid, tree in enumerate(trees):
        nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
        for nid, node in nodes_dict.items():
            node_id_map[id(node)] = idx
            node.tid = tid
            idx += 1
    total_nodes = idx
    _get = node_id_map.__getitem__

    def _sibling(n):
        pc = n.parent._children
        return pc[1] if pc[0] is n else pc[0]

    layers_up = batch_topological_sort(trees, order="up")
    up_layers = []
    _np_int64 = np.int64
    for layer in layers_up:
        leaf_idx, leaf_cells, int_idx, int_c0, int_c1 = [], [], [], [], []
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

    _buf3 = cp.zeros((3 * total_nodes, h, w), dtype=cp.float32)
    all_U  = _buf3[0*total_nodes:1*total_nodes]
    all_U_ = _buf3[1*total_nodes:2*total_nodes]
    all__U = _buf3[2*total_nodes:3*total_nodes]
    base_U  = all_U.data.ptr
    base_U_ = all_U_.data.ptr
    base__U = all__U.data.ptr

    _matmul = util.batch_log_matmul_cuda()
    _add_matmul = util.batch_add_log_matmul_cuda()
    _matadd = util.batch_matadd_cuda()

    for layer_info in up_layers:
        if layer_info[0] == 'leaf':
            _, indices, cell_ids = layer_info
            nb = len(indices)
            all_U[indices] = probs[cell_ids]
            ptr_u  = base_U  + indices * stride
            ptr_u_ = base_U_ + indices * stride
            ptr__u = base__U + indices * stride
            s._launch_batched_matmul(_matmul, ptr_u, s.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
            s._launch_batched_matmul(_matmul, ptr_u, s.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)
        else:
            _, indices, c0, c1 = layer_info
            nb = len(indices)
            ptr_c0_u_ = base_U_ + c0 * stride
            ptr_c1_u_ = base_U_ + c1 * stride
            ptr_u  = base_U  + indices * stride
            ptr_u_ = base_U_ + indices * stride
            ptr__u = base__U + indices * stride
            s._launch_batched_add_matmul(_add_matmul, ptr_c0_u_, ptr_c1_u_, s.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
            s._launch_batched_matadd(_matadd, ptr_c0_u_, ptr_c1_u_, ptr_u, nb, h, w)
            s._launch_batched_matmul(_matmul, ptr_u, s.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)

    # Extract root scores: max over 3 root genotype states, sum over sites
    root_list = []
    for tid, tree in enumerate(trees):
        nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
        for nid, node in nodes_dict.items():
            if node.is_root():
                root_list.append(_get(id(node)))
    root_gpu = cp.asarray(np.array(root_list, dtype=_np_int64))
    root_U = all_U[root_gpu]  # (n_trees, h, w)
    root_scores = cp.zeros((len(trees), 3, h), dtype=cp.float32)
    root_scores[:, 0, :] = root_U[:, :, s.index_gt(0, 2)]
    root_scores[:, 1, :] = root_U[:, :, s.index_gt(1, 1)]
    root_scores[:, 2, :] = root_U[:, :, s.index_gt(2, 0)]
    # Per-site max over 3 states, then sum over sites
    root_likelihoods = root_scores.max(axis=1).sum(axis=-1, dtype=cp.float64)
    return root_likelihoods


# ── Main analysis ──
DATASETS = [
    ("autoresearch/sim_data_reads.csv", "100c_100s"),
    ("autoresearch/test_50c_50s_reads.csv", "50c_50s"),
    ("autoresearch/test_100c_200s_reads.csv", "100c_200s"),
]

for data_path, tag in DATASETS:
    reads, cell_names, _ = scna.util.read_csv(data_path)
    n_sites, n_cells, _ = reads.shape
    start_tree, _ = external.infer_scistree2_tree(reads, cell_names=cell_names)
    start_tree = util.relabel(start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)})
    cn_avg = estimate_copy_number(reads[:, :, -1], start_tree)
    s = ScisTreeCNA(CN_MAX=5, CN_MIN=1, LAMBDA_C=cn_avg, LAMBDA_S=1, LAMBDA_T=2*n_cells-1, verbose=False)
    probs = s.init_prob_leaves_gpu(reads, ado=0.1, seqerr=0.01, cnerr=0.05, af=0.5)

    print(f"\n{'='*60}")
    print(f"Dataset: {tag} ({n_cells} cells, {n_sites} sites)")
    print(f"{'='*60}")

    # Run 3 NNI rounds, compare rankings each round
    tree = start_tree
    for round_i in range(3):
        # Generate candidates
        candidates = []
        nodes_dict = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
        for nid, nd in nodes_dict.items():
            if nd.is_leaf():
                continue
            ch = nd._children
            if len(ch) == 2 and not ch[0].is_leaf() and not ch[1].is_leaf():
                t1 = struct_copy_tree(tree)
                p1, p2 = t1[nid]._children
                lc1, lc2 = p1._children; rc1, rc2 = p2._children
                p1.remove_child(lc1); p2.remove_child(rc1)
                lc1.set_parent(p2); rc1.set_parent(p1)
                p1.add_child(rc1); p2.add_child(lc1)
                candidates.append(t1)
                t2 = struct_copy_tree(tree)
                p1, p2 = t2[nid]._children
                lc1, lc2 = p1._children; rc1, rc2 = p2._children
                p1.remove_child(lc1); p2.remove_child(rc2)
                lc1.set_parent(p2); rc2.set_parent(p1)
                p1.add_child(rc2); p2.add_child(lc1)
                candidates.append(t2)
            if not nd.is_root():
                t1 = struct_copy_tree(tree)
                nd1 = t1[nid]; par = nd1.parent
                pc = par._children; sib = pc[1] if pc[0] is nd1 else pc[0]
                c1, c2 = nd1._children
                nd1.remove_child(c1); par.remove_child(sib)
                nd1.add_child(sib); sib.set_parent(nd1)
                par.add_child(c1); c1.set_parent(par)
                candidates.append(t1)
                t2 = struct_copy_tree(tree)
                nd2 = t2[nid]; par = nd2.parent
                pc = par._children; sib = pc[1] if pc[0] is nd2 else pc[0]
                c1, c2 = nd2._children
                nd2.remove_child(c2); par.remove_child(sib)
                nd2.add_child(sib); sib.set_parent(nd2)
                par.add_child(c2); c2.set_parent(par)
                candidates.append(t2)

        best_tree = struct_copy_tree(tree)
        all_trees = [best_tree] + candidates

        # Compute full scores
        full_scores = s.marginal_evaluate_dp_batch(probs, all_trees, batch_size=256)
        cp.cuda.Stream.null.synchronize()

        # Compute root-only scores
        all_trees2 = [struct_copy_tree(tree)] + [struct_copy_tree(c) for c in candidates]
        root_scores = compute_root_only_scores(s, probs, all_trees2)
        cp.cuda.Stream.null.synchronize()

        full_rank = cp.argsort(-full_scores).get()
        root_rank = cp.argsort(-root_scores).get()

        full_best = full_rank[0]
        root_best = root_rank[0]

        # Check top-K overlap
        for K in [1, 3, 5, 10, 20]:
            top_k_full = set(full_rank[:K].tolist())
            top_k_root = set(root_rank[:K].tolist())
            overlap = len(top_k_full & top_k_root)
            print(f"  Round {round_i}, top-{K}: overlap={overlap}/{K} ({overlap/K*100:.0f}%)", end="")
            if K == 1:
                match = "✅" if full_best == root_best else "❌"
                print(f"  full_best={full_best} root_best={root_best} {match}", end="")
            print()

        # Rank correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(full_scores.get(), root_scores.get())
        print(f"  Spearman rank correlation: {corr:.4f}")

        # Move to next round using full scoring winner
        tree = all_trees[int(full_best)]
