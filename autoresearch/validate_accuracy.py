"""Run inference on multiple datasets and compare results.
Used to validate that optimizations don't change results."""
import time
import numpy as np
import cupy as cp
import pickle
import scistreecna as scna

DATASETS = [
    {"tag": "50c_50s", "max_iter": 5},
    {"tag": "100c_100s_seed123", "max_iter": 5},
    {"tag": "100c_200s", "max_iter": 5},
    {"tag": "200c_100s", "max_iter": 3},
]

INFER_PARAMS_BASE = dict(
    ado=0.1, seq_error=0.01, cn_noise=0.05,
    cn_min=1, cn_max=5,
    tree_batch_size=128, node_batch_size=256,
)

results = []

for ds in DATASETS:
    tag = ds["tag"]
    print(f"\n{'='*50}")
    print(f"Dataset: {tag}")
    print(f"{'='*50}")

    reads, cell_names, _ = scna.util.read_csv(f"autoresearch/test_{tag}_reads.csv")
    with open(f"autoresearch/test_{tag}_tree.pkl", "rb") as f:
        true_tree = pickle.load(f)
    true_geno = np.loadtxt(f"autoresearch/test_{tag}_tg.txt", dtype=int)

    print(f"Data: {reads.shape}")

    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    tree, geno = scna.infer(
        reads, cell_names=cell_names,
        max_iter=ds["max_iter"],
        verbose=True,
        verbose_mode="min",
        **INFER_PARAMS_BASE,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0

    tree_acc = scna.util.tree_accuracy(true_tree, tree)
    geno_acc = scna.util.genotype_accuarcy(true_geno, geno)

    # Re-evaluate likelihood
    n_sites, n_cells, _ = reads.shape
    start_tree, _ = scna.external.infer_scistree2_tree(reads, cell_names=cell_names)
    start_tree = scna.util.relabel(start_tree, name_map={name: str(i) for i, name in enumerate(cell_names)})
    cn_avg = scna.scistreecna.estimate_copy_number(reads[:, :, -1], start_tree)
    s = scna.scistreecna.ScisTreeCNA(
        CN_MAX=5, CN_MIN=1, LAMBDA_C=cn_avg, LAMBDA_S=1, LAMBDA_T=2*n_cells-1, verbose=False
    )
    probs = s.init_prob_leaves_gpu(reads, ado=0.1, seqerr=0.01, cnerr=0.05, af=0.5)
    eval_tree = scna.util.relabel(tree, name_map={name: str(i) for i, name in enumerate(cell_names)})
    lh, _ = s.marginal_evaluate_dp(probs, eval_tree)
    likelihood = float(lh)

    r = {
        "tag": tag,
        "time": elapsed,
        "likelihood": likelihood,
        "tree_acc": tree_acc,
        "geno_acc": geno_acc,
    }
    results.append(r)
    print(f"Time: {elapsed:.4f}s")
    print(f"Likelihood: {likelihood:.6f}")
    print(f"Tree accuracy: {tree_acc:.4f}")
    print(f"Genotype accuracy: {geno_acc:.4f}")

print(f"\n{'='*60}")
print(f"VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"{'Tag':<25} {'Time':>8} {'Likelihood':>14} {'TreeAcc':>8} {'GenoAcc':>8}")
for r in results:
    print(f"{r['tag']:<25} {r['time']:>8.2f} {r['likelihood']:>14.4f} {r['tree_acc']:>8.4f} {r['geno_acc']:>8.4f}")
print(f"{'='*60}")
