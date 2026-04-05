"""Benchmark comparison: run inference on multiple datasets, report time and accuracy.
Run this script on both branches (main vs gpu-optimization) and compare."""
import time
import numpy as np
import cupy as cp
import pickle
import scistreecna as scna

DATASETS = [
    {"tag": "50c_50s", "path": "autoresearch/test_50c_50s", "max_iter": 10},
    {"tag": "100c_100s", "path": "autoresearch/sim_data", "max_iter": 10},
    {"tag": "100c_100s_v2", "path": "autoresearch/test_100c_100s_seed123", "max_iter": 10},
    {"tag": "100c_200s", "path": "autoresearch/test_100c_200s", "max_iter": 10},
    {"tag": "200c_100s", "path": "autoresearch/test_200c_100s", "max_iter": 5},
]

INFER_PARAMS = dict(
    ado=0.1, seq_error=0.01, cn_noise=0.05,
    cn_min=1, cn_max=5,
    tree_batch_size=128, node_batch_size=256,
    verbose=True, verbose_mode="min",
)

N_RUNS = 3  # median of N runs

results = []

for ds in DATASETS:
    tag = ds["tag"]
    reads_path = ds["path"] + "_reads.csv"
    tree_path = ds["path"] + "_tree.pkl"
    geno_path = ds["path"] + "_tg.txt"

    print(f"\n{'='*60}")
    print(f"Dataset: {tag}")
    print(f"{'='*60}")

    reads, cell_names, _ = scna.util.read_csv(reads_path)
    with open(tree_path, "rb") as f:
        true_tree = pickle.load(f)
    true_geno = np.loadtxt(geno_path, dtype=int)
    n_sites, n_cells, _ = reads.shape
    print(f"Data: {reads.shape} ({n_cells} cells, {n_sites} sites)")

    # Warmup
    _ = cp.zeros(10)
    cp.cuda.Stream.null.synchronize()

    times = []
    last_tree = None
    last_geno = None
    for run_i in range(N_RUNS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        tree, geno = scna.infer(
            reads, cell_names=cell_names,
            max_iter=ds["max_iter"],
            **INFER_PARAMS,
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        last_tree = tree
        last_geno = geno
        print(f"  Run {run_i+1}: {elapsed:.4f}s")

    median_time = np.median(times)
    tree_acc = scna.util.tree_accuracy(true_tree, last_tree)
    geno_acc = scna.util.genotype_accuarcy(true_geno, last_geno)

    # Re-evaluate likelihood
    start_tree, _ = scna.external.infer_scistree2_tree(reads, cell_names=cell_names)
    start_tree = scna.util.relabel(start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)})
    cn_avg = scna.scistreecna.estimate_copy_number(reads[:, :, -1], start_tree)
    s = scna.scistreecna.ScisTreeCNA(
        CN_MAX=5, CN_MIN=1, LAMBDA_C=cn_avg, LAMBDA_S=1, LAMBDA_T=2*n_cells-1, verbose=False
    )
    probs = s.init_prob_leaves_gpu(reads, ado=0.1, seqerr=0.01, cnerr=0.05, af=0.5)
    eval_tree = scna.util.relabel(last_tree, name_map={name: str(i) for i, name in enumerate(cell_names)})
    lh, _ = s.marginal_evaluate_dp(probs, eval_tree)
    likelihood = float(lh)

    r = {
        "tag": tag,
        "cells": n_cells,
        "sites": n_sites,
        "max_iter": ds["max_iter"],
        "median_time": median_time,
        "all_times": times,
        "likelihood": likelihood,
        "tree_acc": tree_acc,
        "geno_acc": geno_acc,
    }
    results.append(r)

print(f"\n\n{'='*80}")
print(f"BENCHMARK RESULTS")
print(f"{'='*80}")
print(f"{'Tag':<16} {'Cells':>5} {'Sites':>5} {'Iter':>4} {'Time(s)':>8} {'Likelihood':>14} {'TreeAcc':>8} {'GenoAcc':>8}")
print(f"{'-'*80}")
for r in results:
    print(f"{r['tag']:<16} {r['cells']:>5} {r['sites']:>5} {r['max_iter']:>4} {r['median_time']:>8.3f} {r['likelihood']:>14.2f} {r['tree_acc']:>8.4f} {r['geno_acc']:>8.4f}")
print(f"{'='*80}")
