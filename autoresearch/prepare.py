"""
ScisTreeCNA AutoResearch Evaluation Harness
============================================
DO NOT MODIFY THIS FILE. This is the fixed evaluation script.

Runs inference on the example dataset, measures timing, and checks correctness
constraints. Outputs a single RESULT line for automated parsing.

Based on Karpathy's autoresearch pattern.
"""
import time
import sys
import numpy as np
import cupy as cp
import pickle
import scistreecna as scna

# ── Fixed evaluation parameters ──────────────────────────────────────────────
DATA_PATH = "autoresearch/sim_data_reads.csv"
TRUE_TREE_PATH = "autoresearch/sim_data_tree.pkl"
TRUE_GENO_PATH = "autoresearch/sim_data_tg.txt"

INFER_PARAMS = dict(
    ado=0.1,
    seq_error=0.01,
    cn_noise=0.05,
    cn_min=1,
    cn_max=5,
    tree_batch_size=128,
    node_batch_size=256,
    max_iter=10,
    verbose=True,
    verbose_mode="all",
)

# ── Correctness constraints ──────────────────────────────────────────────────
# These are set from the baseline run on 100-cell 100-site simulated data
BASELINE_LIKELIHOOD = -15164.22
LIKELIHOOD_TOL = 2.0        # must be within 2.0 of baseline
MIN_TREE_ACCURACY = 0.0     # tree_accuracy depends on tree topology metric, can be 0 for sim data
MIN_GENO_ACCURACY = 0.98    # must be >= this

# ── Number of timed runs ─────────────────────────────────────────────────────
N_WARMUP = 1
N_TIMED = 3


def run_once(reads, cell_names):
    """Run a single inference and return (time, tree, geno)."""
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    tree, geno = scna.infer(reads, cell_names=cell_names, **INFER_PARAMS)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0
    return elapsed, tree, geno


def evaluate():
    # Load data
    reads, cell_names, site_names = scna.util.read_csv(DATA_PATH)
    print(f"Data: {reads.shape} (sites={reads.shape[0]}, cells={reads.shape[1]})")

    # Load ground truth
    with open(TRUE_TREE_PATH, "rb") as f:
        true_tree = pickle.load(f)
    true_genotype = np.loadtxt(TRUE_GENO_PATH, dtype=int)

    # Warmup
    for i in range(N_WARMUP):
        print(f"\n--- Warmup {i+1}/{N_WARMUP} ---")
        run_once(reads, cell_names)

    # Timed runs
    times = []
    last_tree = None
    last_geno = None
    for i in range(N_TIMED):
        print(f"\n--- Timed run {i+1}/{N_TIMED} ---")
        elapsed, tree, geno = run_once(reads, cell_names)
        times.append(elapsed)
        last_tree = tree
        last_geno = geno
        print(f"Run {i+1}: {elapsed:.4f}s")

    # Compute metrics
    total_time = np.median(times)
    tree_acc = scna.util.tree_accuracy(true_tree, last_tree)
    geno_acc = scna.util.genotype_accuarcy(true_genotype, last_geno)

    # Extract likelihood from the last run (best from local search)
    # Re-evaluate with correct parameters derived from data
    n_sites, n_cells, _ = reads.shape
    # Compute cn_avg the same way infer() does
    start_tree_eval, _ = scna.external.infer_scistree2_tree(reads, cell_names=cell_names)
    start_tree_eval = scna.util.relabel(
        start_tree_eval, name_map={name: str(i) for i, name in enumerate(cell_names)}
    )
    cn_avg = scna.scistreecna.estimate_copy_number(reads[:, :, -1], start_tree_eval)
    s = scna.scistreecna.ScisTreeCNA(
        CN_MAX=5, CN_MIN=1, LAMBDA_C=cn_avg, LAMBDA_S=1, LAMBDA_T=2*n_cells-1, verbose=False
    )
    probs = s.init_prob_leaves_gpu(reads, ado=0.1, seqerr=0.01, cnerr=0.05, af=0.5)
    # Relabel tree back to numeric for evaluation
    name_map = {name: str(i) for i, name in enumerate(cell_names)}
    eval_tree = scna.util.relabel(last_tree, name_map=name_map)
    likelihood_val, _ = s.marginal_evaluate_dp(probs, eval_tree)
    likelihood = float(likelihood_val)

    # Check constraints
    lh_ok = abs(likelihood - BASELINE_LIKELIHOOD) <= LIKELIHOOD_TOL
    tree_ok = tree_acc >= MIN_TREE_ACCURACY
    geno_ok = geno_acc >= MIN_GENO_ACCURACY
    status = "PASS" if (lh_ok and tree_ok and geno_ok) else "FAIL"

    # Print detailed results
    print(f"\n{'='*60}")
    print(f"AUTORESEARCH EVALUATION")
    print(f"{'='*60}")
    print(f"Median time ({N_TIMED} runs): {total_time:.4f}s")
    print(f"All times: {[f'{t:.4f}' for t in times]}")
    print(f"Likelihood:        {likelihood:.4f} (baseline: {BASELINE_LIKELIHOOD}, tol: {LIKELIHOOD_TOL}) {'OK' if lh_ok else 'FAIL'}")
    print(f"Tree accuracy:     {tree_acc:.4f} (min: {MIN_TREE_ACCURACY}) {'OK' if tree_ok else 'FAIL'}")
    print(f"Genotype accuracy: {geno_acc:.4f} (min: {MIN_GENO_ACCURACY}) {'OK' if geno_ok else 'FAIL'}")
    print(f"Status: {status}")
    print(f"{'='*60}")

    # Machine-readable result line
    print(f"RESULT: total_time_s={total_time:.4f} likelihood={likelihood:.2f} tree_accuracy={tree_acc:.4f} genotype_accuracy={geno_acc:.4f} status={status}")

    return status == "PASS"


if __name__ == "__main__":
    success = evaluate()
    sys.exit(0 if success else 1)
