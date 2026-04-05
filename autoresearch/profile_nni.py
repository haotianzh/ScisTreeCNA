"""Profile a single NNI round to find where time is spent."""
import time
import numpy as np
import cupy as cp
import scistreecna as scna
from scistreecna.scistreecna import ScisTreeCNA, TreeBatchLoader
from scistreecna.scistreecna import estimate_copy_number
from scistreecna.base.tree import struct_copy_tree
from scistreecna import util, external
from scistreecna.topological_sort import batch_topological_sort

# Load data
reads, cell_names, _ = scna.util.read_csv('autoresearch/sim_data_reads.csv')
n_sites, n_cells, _ = reads.shape
start_tree, _ = external.infer_scistree2_tree(reads, cell_names=cell_names)
start_tree = util.relabel(start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)})
cn_avg = estimate_copy_number(reads[:, :, -1], start_tree)

s = ScisTreeCNA(CN_MAX=5, CN_MIN=1, LAMBDA_C=cn_avg, LAMBDA_S=1, LAMBDA_T=2*n_cells-1, verbose=False)
probs = s.init_prob_leaves_gpu(reads, ado=0.1, seqerr=0.01, cnerr=0.05, af=0.5)
tree = start_tree

print(f"Data: {n_cells} cells, {n_sites} sites, N={s.N} states")
print(f"Tree: {len(tree.get_all_nodes())} nodes")
print()

# ── Profile NNI candidate generation ──
t0 = time.time()
candidates = []
for node in tree.get_all_nodes():
    sw = not tree[node].is_leaf()
    for child in tree[node].get_children():
        if child.is_leaf(): sw = False
    if sw:
        t1 = struct_copy_tree(tree)
        p1, p2 = t1[node].get_children()
        lc1, lc2 = p1.get_children()
        rc1, rc2 = p2.get_children()
        p1.remove_child(lc1); p2.remove_child(rc1)
        lc1.set_parent(p2); rc1.set_parent(p1)
        p1.add_child(rc1); p2.add_child(lc1)
        candidates.append(t1)
        t2 = struct_copy_tree(tree)
        p1, p2 = t2[node].get_children()
        lc1, lc2 = p1.get_children()
        rc1, rc2 = p2.get_children()
        p1.remove_child(lc1); p2.remove_child(rc2)
        lc1.set_parent(p2); rc2.set_parent(p1)
        p1.add_child(rc2); p2.add_child(lc1)
        candidates.append(t2)
for node in tree.get_all_nodes():
    sw = not tree[node].is_leaf() and not tree[node].is_root()
    if sw:
        t1 = struct_copy_tree(tree)
        sib = t1[node].get_siblings()[0]
        c1, c2 = t1[node].get_children()
        t1[node].remove_child(c1); t1[node].parent.remove_child(sib)
        t1[node].add_child(sib); sib.set_parent(t1[node])
        t1[node].parent.add_child(c1); c1.set_parent(t1[node].parent)
        candidates.append(t1)
        t2 = struct_copy_tree(tree)
        sib = t2[node].get_siblings()[0]
        c1, c2 = t2[node].get_children()
        t2[node].remove_child(c2); t2[node].parent.remove_child(sib)
        t2[node].add_child(sib); sib.set_parent(t2[node])
        t2[node].parent.add_child(c2); c2.set_parent(t2[node].parent)
        candidates.append(t2)
t_cand = time.time() - t0
print(f"1. Candidate generation: {t_cand*1000:.1f} ms ({len(candidates)} candidates)")

# ── Profile batch eval ──
best_tree = struct_copy_tree(tree)
all_trees = [best_tree] + candidates
tree_batch_size = 128

loader = TreeBatchLoader(all_trees, batch_size=tree_batch_size)
t_total_eval = 0
t_total_topo = 0
t_total_gpu = 0
n_batches = 0

for bi, trees_batch in enumerate(loader()):
    n_batches += 1

    # Phase 1: topology
    t0 = time.time()
    h, w = probs.shape[1:]
    stride = h * w * 4
    node_id_map = {}
    idx = 0
    for tid, tr in enumerate(trees_batch):
        nodes_dict = tr._nodes if hasattr(tr, '_nodes') else tr.get_all_nodes()
        for nid, node in nodes_dict.items():
            node_id_map[id(node)] = idx
            node.tid = tid
            idx += 1
    total_nodes = idx

    layers_up = batch_topological_sort(trees_batch, order="up")
    up_layers = []
    for layer in layers_up:
        leaves = [n for n in layer if n.is_leaf()]
        internals = [n for n in layer if not n.is_leaf()]
        if leaves:
            idx_l = cp.array([node_id_map[id(n)] for n in leaves], dtype=cp.int64)
            cells = cp.array([int(n.name) for n in leaves], dtype=cp.int64)
            up_layers.append(('leaf', idx_l, cells))
        if internals:
            idx_i = cp.array([node_id_map[id(n)] for n in internals], dtype=cp.int64)
            c0 = cp.array([node_id_map[id(n.get_children()[0])] for n in internals], dtype=cp.int64)
            c1 = cp.array([node_id_map[id(n.get_children()[1])] for n in internals], dtype=cp.int64)
            up_layers.append(('internal', idx_i, c0, c1))
    t_topo = time.time() - t0
    t_total_topo += t_topo

    # Full eval
    t0 = time.time()
    lh = s.marginal_evaluate_dp_batch(probs, trees_batch, batch_size=256)
    cp.cuda.Stream.null.synchronize()
    t_eval = time.time() - t0
    t_total_eval += t_eval

print(f"\n2. Batch evaluation ({n_batches} batches, {len(all_trees)} trees):")
print(f"   Topology pre-computation: {t_total_topo*1000:.1f} ms")
print(f"   Full eval (incl. topo):   {t_total_eval*1000:.1f} ms")
print(f"   GPU-only (eval - topo):   {(t_total_eval - t_total_topo)*1000:.1f} ms")

# ── Profile single-tree eval at end ──
t0 = time.time()
ml, idx = s.marginal_evaluate_dp(probs, struct_copy_tree(tree))
cp.cuda.Stream.null.synchronize()
t_single = time.time() - t0
print(f"\n3. Single-tree marginal_evaluate_dp: {t_single*1000:.1f} ms")

# ── Total ──
total = t_cand + t_total_eval + t_single
print(f"\n{'='*50}")
print(f"TOTAL per NNI round: {total*1000:.0f} ms")
print(f"  Candidate gen:  {t_cand*1000:.0f} ms ({t_cand/total*100:.0f}%)")
print(f"  Batch eval:     {t_total_eval*1000:.0f} ms ({t_total_eval/total*100:.0f}%)")
print(f"  Single eval:    {t_single*1000:.0f} ms ({t_single/total*100:.0f}%)")
print(f"{'='*50}")
