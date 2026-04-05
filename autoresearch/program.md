# ScisTreeCNA AutoResearch Program

You are an autonomous GPU optimization researcher. Your single objective is to **minimize the total execution time** of ScisTreeCNA's cell lineage tree inference while maintaining numerical correctness.

## Setup (one-time)

1. Create a fresh branch: `git checkout -b autoresearch/<tag>`
2. Read and understand these files:
   - `README.md` — project overview
   - `OPTIMIZATION_REPORT.md` — prior optimizations already applied
   - `autoresearch/prepare.py` — **READ-ONLY** benchmark harness
   - `autoresearch/results.tsv` — experiment log
3. Verify baseline: `python autoresearch/prepare.py`
4. Confirm the baseline results match expected values before starting

## Metric

**Primary metric: `total_time_s`** — lower is better.

**Correctness constraints (MUST NOT change):**
- `likelihood` must remain within 2.0 of baseline (-15164.22)
- `genotype_accuracy` must remain >= 0.98

If ANY correctness constraint is violated, the experiment is **automatically rejected**.

## Files You May Edit

| File | What it contains |
|------|-----------------|
| `scistreecna/scistreecna.py` | Main inference engine: NNI search, batch DP evaluation, kernel launch logic |
| `scistreecna/util/kernels.py` | CUDA RawKernel source code (log_matmul, batch kernels, fused kernels) |
| `scistreecna/util/__init__.py` | Kernel wrapper functions |
| `scistreecna/util/util_common.py` | Common utility functions |
| `scistreecna/base/tree.py` | Tree data structure, struct_copy_tree |
| `scistreecna/topological_sort.py` | Batch topological sort |
| `scistreecna/transition_solver.py` | Transition probability solver |
| `scistreecna/cn_estimate.py` | Copy number estimation |

## Files You Must NOT Edit

| File | Why |
|------|-----|
| `autoresearch/prepare.py` | Fixed evaluation harness — ensures fair comparison |
| `examples/*` | Test data |
| `autoresearch/program.md` | This file |
| `autoresearch/results.tsv` | Only append to this file, never delete rows |

## Experiment Loop

Repeat these steps indefinitely until interrupted:

### 1. Plan
Review `autoresearch/results.tsv` and the current code. Think about what optimization to try next. Consider:
- Python overhead reduction (eliminate loops, cache computations)
- CUDA kernel optimization (shared memory, register usage, thread occupancy, fused kernels)
- Memory allocation patterns (reduce allocations, pre-allocate buffers, single contiguous blocks)
- Algorithmic improvements (avoid redundant work, better data structures)
- CuPy-specific optimizations (in-place operations, avoid unnecessary copies)

### 2. Implement
Make a **single, focused change** to one or two files. Keep changes small and testable. Do not combine multiple unrelated ideas in one experiment.

### 3. Commit
```bash
git add -A && git commit -m "exp: <short description of what you changed>"
```

### 4. Run
```bash
python autoresearch/prepare.py > autoresearch/run.log 2>&1
```
- Kill the run if it exceeds 60 seconds (something is probably broken)
- If it crashes, read the stack trace, fix the bug, and re-run

### 5. Evaluate
```bash
grep "^RESULT:" autoresearch/run.log
```
The output format is:
```
RESULT: total_time_s=X.XXXX likelihood=-XXXXX.XX tree_accuracy=X.XXXX genotype_accuracy=X.XXXX status=PASS/FAIL
```

### 6. Record
Append the result to `autoresearch/results.tsv`:
```
<experiment_number>\t<commit_hash>\t<description>\t<total_time_s>\t<likelihood>\t<tree_acc>\t<geno_acc>\t<status>
```

### 7. Keep or Discard
- **If `status=PASS` AND `total_time_s` improved**: Keep the commit. This is the new baseline.
- **If `status=FAIL` OR `total_time_s` did not improve**: Revert:
  ```bash
  git reset --hard HEAD~1
  ```

### 8. Loop
Go back to step 1. **NEVER STOP.** Continue until manually interrupted.

## Optimization Ideas to Explore

These are starting points. You should generate your own ideas too.

### High Priority
- Remove redundant `struct_copy_tree` calls in eval path (line ~700 in scistreecna.py)
- Cache `tran_prob_mutation_free.T` in `__init__` instead of recomputing per eval call
- Merge 4 separate `cp.zeros()` allocations into single contiguous block
- Pre-allocate GPU buffers across NNI iterations (reuse instead of re-allocating)

### Medium Priority
- Optimize `batch_log_3vecdot` kernel: cache intermediate values in registers instead of re-reading global memory
- Fix O(n^2) `list.remove()` in `batch_topological_sort`
- Use CUDA streams for overlapping independent kernel launches
- Reduce Python list comprehension overhead in topology pre-computation

### Experimental
- Half-precision (float16) computation where numerical stability allows
- Warp-level primitives for small reductions
- Custom memory pool to avoid CuPy allocator overhead
- Overlap NNI candidate generation with GPU evaluation

## Rules

1. **One idea per experiment.** Small, focused changes are easier to evaluate.
2. **Correctness first.** Never sacrifice accuracy for speed. If in doubt, check.
3. **Measure everything.** Don't guess if something is faster — measure it.
4. **Simplicity criterion.** Prefer elegant improvements over complex marginal gains.
5. **Never install new packages.** Only use what's in `pyproject.toml` + CuPy.
6. **Never modify the evaluation harness** (`autoresearch/prepare.py`).
7. **NEVER STOP.** Keep iterating until manually interrupted.
