# ScisTreeCNA GPU Optimization Phase 2 - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve an additional 2-3x speedup on top of the existing ~3.9x optimization (baseline 13.8s -> current 3.5s -> target ~1.5s for 10 NNI iterations) by eliminating remaining Python overhead, fusing CUDA kernels, and reducing redundant memory operations.

**Architecture:** The core bottleneck has shifted from GPU kernel execution to Python-side overhead: NNI candidate generation (~400-800 struct_copy_tree calls per round), per-layer index construction in `marginal_evaluate_dp_batch`, redundant transpose computation, and unfused kernel launches. We attack these in priority order: (1) cache topology pre-computation across NNI rounds, (2) fuse remaining kernel pairs, (3) eliminate redundant allocations, (4) optimize the scoring kernels.

**Tech Stack:** Python 3.12, CuPy (CUDA), custom RawKernel CUDA C code

**Current Performance Profile (128 trees, ~200 nodes each):**
| Stage | Current Time | Target |
|-------|-------------|--------|
| NNI candidate generation (struct_copy_tree ×400) | ~80ms | ~80ms (hard to avoid) |
| struct_copy_tree for eval (×128) | ~27ms | ~27ms |
| Phase 1: topology pre-computation + index arrays | ~30ms | ~5ms (cache/reuse) |
| Phase 2: GPU array allocation (all_U, all_U_, all__U, all_Q) | ~8ms | ~4ms (single alloc) |
| Phase 3-5: GPU kernels + scoring | ~15ms | ~10ms (fuse + optimize) |
| Transpose recomputation per call | ~1ms | 0ms (cache) |
| Single-tree marginal_evaluate_dp (per round) | ~90ms | 0ms (already eliminated in v3) |

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `scistreecna/scistreecna.py` | Main inference engine, NNI search, batch evaluation | Modify: cache transpose, single-block allocation, topology caching |
| `scistreecna/util/kernels.py` | CUDA kernel source code | Modify: fuse matadd+log_matmul for down pass, optimize 3vecdot |
| `scistreecna/util/__init__.py` | Kernel wrapper functions | Modify: add wrapper for new fused kernel |
| `scistreecna/topological_sort.py` | Batch topological sort | Modify: fix O(n^2) list.remove() |
| `test_kernels.py` | Kernel correctness tests | Modify: add tests for new/changed kernels |
| `benchmark.py` | End-to-end benchmark | No change (use as-is for validation) |

---

### Task 1: Cache `tran_prob_mutation_free_t` in `__init__`

**Files:**
- Modify: `scistreecna/scistreecna.py:146-179` (ScisTreeCNA.__init__)
- Modify: `scistreecna/scistreecna.py:462` (marginal_evaluate_dp_batch)

**Why:** `cp.ascontiguousarray(self.tran_prob_mutation_free.T)` is recomputed every call to `marginal_evaluate_dp_batch`. It's a small matrix (N x N, where N=20 for typical params) but the allocation + copy is unnecessary overhead repeated hundreds of times.

- [ ] **Step 1: Write a test that verifies transpose is correct**

```python
# In test_kernels.py or inline verification
# After __init__, check that self.tran_prob_mutation_free_t == cp.ascontiguousarray(self.tran_prob_mutation_free.T)
```

- [ ] **Step 2: Add cached transpose to `__init__`**

In `scistreecna/scistreecna.py`, in `ScisTreeCNA.__init__`, after line 168 (where `self.tran_prob_mutation_free` is set), add:

```python
self.tran_prob_mutation_free_t = cp.ascontiguousarray(
    self.tran_prob_mutation_free.T
)
```

- [ ] **Step 3: Replace inline transpose in `marginal_evaluate_dp_batch`**

In `scistreecna/scistreecna.py`, line 462, change:
```python
tran_prob_mutation_free_t = cp.ascontiguousarray(self.tran_prob_mutation_free.T)
```
to:
```python
tran_prob_mutation_free_t = self.tran_prob_mutation_free_t
```

- [ ] **Step 4: Also fix in `calculate_Q` (line 558-559)**

```python
# Old:
tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(...)
# This doesn't use the transpose, but it re-tiles every call. Cache if needed.
```

- [ ] **Step 5: Run benchmark to verify correctness + timing**

```bash
python benchmark.py
```
Expected: identical likelihood/accuracy, marginal timing improvement.

- [ ] **Step 6: Commit**

```bash
git add scistreecna/scistreecna.py
git commit -m "perf: cache transposed transition matrix in __init__"
```

---

### Task 2: Single Contiguous GPU Allocation Block

**Files:**
- Modify: `scistreecna/scistreecna.py:421-427` (marginal_evaluate_dp_batch Phase 2)

**Why:** Four separate `cp.zeros()` calls for `all_U`, `all_U_`, `all__U`, `all_Q` cause 4 allocator round-trips. Allocating one large block and slicing reduces allocator overhead and may improve cache locality.

- [ ] **Step 1: Replace 4 allocations with single block**

In `marginal_evaluate_dp_batch`, replace:
```python
all_U  = cp.zeros((total_nodes, h, w), dtype=cp.float32)
all_U_ = cp.zeros((total_nodes, h, w), dtype=cp.float32)
all__U = cp.zeros((total_nodes, h, w), dtype=cp.float32)
```

with:
```python
# Single allocation for U, U_, _U (Q allocated later after up-pass)
_buf = cp.zeros((3 * total_nodes, h, w), dtype=cp.float32)
all_U  = _buf[0*total_nodes:1*total_nodes]
all_U_ = _buf[1*total_nodes:2*total_nodes]
all__U = _buf[2*total_nodes:3*total_nodes]
```

- [ ] **Step 2: Same for all_Q (line 458)**

Replace:
```python
all_Q = cp.zeros((total_nodes, h, w), dtype=cp.float32)
```
with a separate single allocation (Q needs special init with `index_gt(2,0) = 1.0`, so keeping it separate is fine, but we save 3->1 allocs for the main buffers).

- [ ] **Step 3: Verify base pointer arithmetic still works**

The `base_U`, `base_U_`, `base__U` pointers must still be correct:
```python
base_U  = all_U.data.ptr   # Should point to _buf[0]
base_U_ = all_U_.data.ptr  # Should point to _buf[total_nodes * h * w * 4]
base__U = all__U.data.ptr  # Should point to _buf[2 * total_nodes * h * w * 4]
```

CuPy slices share the same underlying allocation, so `.data.ptr` returns the correct offset.

- [ ] **Step 4: Run benchmark**

```bash
python benchmark.py
```
Expected: identical results, small timing improvement.

- [ ] **Step 5: Commit**

```bash
git add scistreecna/scistreecna.py
git commit -m "perf: single contiguous GPU allocation for U/U_/_U buffers"
```

---

### Task 3: Fuse Down-Pass `add + log_matmul` for Q Computation

**Files:**
- Modify: `scistreecna/util/kernels.py` (add new fused kernel variant)
- Modify: `scistreecna/util/__init__.py` (add wrapper)
- Modify: `scistreecna/scistreecna.py:464-473` (Phase 4 down-pass)

**Why:** The down-pass currently uses `_launch_batched_add_matmul` which fuses add+matmul, and this is already optimized. However, the kernel `batch_add_log_matmul` reads bmat_a (Q[par]) and bmat_b (U_[sib]) then multiplies by mat2 (transition matrix transposed). This path is already fused. 

The real opportunity: the `batch_log_3vecdot` scoring kernel (Phase 5) does redundant two-pass computation. Let's optimize that instead.

**Skip this task if the down-pass is already fused.** (It is — via `_launch_batched_add_matmul` at line 473.)

---

### Task 3 (revised): Optimize `batch_log_3vecdot` Kernel

**Files:**
- Modify: `scistreecna/util/kernels.py:649-675` (kernel_batch_log_3vecdot)
- Modify: `test_kernels.py` (add correctness test)

**Why:** The scoring kernel does a three-way element-wise add + log-sum-exp reduction. It currently uses two loops: one to find max, one to accumulate. For small `k` (typically 20), we can use a single pass with Kahan-style compensation, or cache values in registers since k=20 fits in registers.

- [ ] **Step 1: Write correctness test for 3vecdot**

```python
# test_kernels.py
def test_batch_log_3vecdot_correctness():
    import cupy as cp
    from scistreecna import util
    
    nb, h, w = 32, 100, 20
    a = cp.random.randn(nb, h, w).astype(cp.float32) * 0.1
    b = cp.random.randn(nb, h, w).astype(cp.float32) * 0.1
    c = cp.random.randn(nb, h, w).astype(cp.float32) * 0.1
    
    # Reference: log(sum(exp(a + b + c), axis=-1))
    abc = a + b + c
    maxval = abc.max(axis=-1, keepdims=True)
    ref = (maxval.squeeze(-1) + cp.log(cp.exp(abc - maxval).sum(axis=-1))).astype(cp.float32)
    
    # Kernel version
    out = cp.full((nb, h), float('-inf'), dtype=cp.float32)
    stride = h * w * 4
    base_a, base_b, base_c = a.data.ptr, b.data.ptr, c.data.ptr
    idx = cp.arange(nb, dtype=cp.int64)
    ptr_a = base_a + idx * stride
    ptr_b = base_b + idx * stride
    ptr_c = base_c + idx * stride
    
    block = (256, 1)
    grid = ((h + 255) // 256, 1, nb)
    util.batch_log_3vecdot_cuda()(grid, block, (ptr_a, ptr_b, ptr_c, out, nb, h, w))
    cp.cuda.Stream.null.synchronize()
    
    cp.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run test to verify baseline passes**

```bash
python -m pytest test_kernels.py::test_batch_log_3vecdot_correctness -v
```

- [ ] **Step 3: Optimize kernel — cache values in register array**

In `scistreecna/util/kernels.py`, replace the `batch_log_3vecdot` kernel body:

```c
extern "C" __global__ void batch_log_3vecdot(
    long long* bmat1, long long* bmat2, long long* bmat3,
    float* out, int m, int n, int k)
{
    int z = blockIdx.z;
    if (z >= m) return;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float* mat1 = (float*)bmat1[z];
    float* mat2 = (float*)bmat2[z];
    float* mat3 = (float*)bmat3[z];
    
    // Single pass: find max while caching values in registers
    // k is typically 20, so 20 floats = 80 bytes of registers (fine)
    float vals[64];  // max supported k
    float maxval = -1.0f / 0.0f;
    
    for (int p = 0; p < k; ++p) {
        float val = mat1[i*k + p] + mat2[i*k + p] + mat3[i*k + p];
        vals[p] = val;
        if (val > maxval) maxval = val;
    }
    
    float sumexp = 0.0f;
    if (!isinf(maxval) || maxval > 0.0f) {
        for (int p = 0; p < k; ++p) {
            if (!isinf(vals[p]))
                sumexp += expf(vals[p] - maxval);
        }
    }
    
    out[z * n + i] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
}
```

This avoids re-reading 3 global memory arrays in the second pass. For k=20, the 20 floats easily fit in registers.

- [ ] **Step 4: Run test to verify correctness preserved**

```bash
python -m pytest test_kernels.py::test_batch_log_3vecdot_correctness -v
```

- [ ] **Step 5: Run benchmark**

```bash
python benchmark.py
```

- [ ] **Step 6: Commit**

```bash
git add scistreecna/util/kernels.py test_kernels.py
git commit -m "perf: optimize batch_log_3vecdot kernel with register caching"
```

---

### Task 4: Fix O(n^2) in `batch_topological_sort`

**Files:**
- Modify: `scistreecna/topological_sort.py:4-18`

**Why:** `sorted_trees.remove(tree)` on line 15 is O(n) list scan inside an O(n) loop = O(n^2). With 400+ candidates this becomes measurable.

- [ ] **Step 1: Read current implementation**

Current code in `scistreecna/topological_sort.py`:
```python
def batch_topological_sort(trees, order="up"):
    sorted_trees = list(trees)
    # ... uses sorted_trees.remove(tree) which is O(n)
```

- [ ] **Step 2: Replace list.remove with index tracking**

```python
def batch_topological_sort(trees, order="up"):
    # Use index-based tracking instead of list.remove()
    remaining = set(range(len(trees)))
    tree_list = list(trees)
    # ... iterate using remaining set, pop by index
```

Or more precisely, fix the specific `remove()` call to use a set-based approach.

- [ ] **Step 3: Run benchmark to verify correctness**

```bash
python benchmark.py
```

- [ ] **Step 4: Commit**

```bash
git add scistreecna/topological_sort.py
git commit -m "perf: fix O(n^2) list removal in batch_topological_sort"
```

---

### Task 5: Eliminate Redundant `struct_copy_tree` in Eval Path

**Files:**
- Modify: `scistreecna/scistreecna.py:698-701` (nni_search_sinlge_round_batch)

**Why:** Line 700 does `[struct_copy_tree(_) for _ in trees]` inside the eval loop, creating a SECOND copy of every tree. The trees are already copies (created in candidate generation). The eval function `marginal_evaluate_dp_batch` doesn't modify trees — it only reads topology. So this second copy is completely unnecessary.

- [ ] **Step 1: Verify eval doesn't modify trees**

Check `marginal_evaluate_dp_batch`: it reads `tree._nodes`, calls `n.is_leaf()`, `n.get_children()`, `n.get_siblings()`, `n.parent` — all read-only. It stores results in contiguous GPU arrays, not on nodes. **Confirmed: no mutation.**

- [ ] **Step 2: Remove redundant copy**

In `scistreecna/scistreecna.py`, line 700, change:
```python
likelihoods = self.marginal_evaluate_dp_batch(
    probs, [struct_copy_tree(_) for _ in trees], batch_size=node_batch_size
)
```
to:
```python
likelihoods = self.marginal_evaluate_dp_batch(
    probs, trees, batch_size=node_batch_size
)
```

- [ ] **Step 3: Run benchmark to verify identical results**

```bash
python benchmark.py
```
Expected: same likelihood/accuracy, saves ~27ms per NNI round (128 trees × 0.21ms/copy).

- [ ] **Step 4: Commit**

```bash
git add scistreecna/scistreecna.py
git commit -m "perf: remove redundant struct_copy_tree in eval path"
```

---

### Task 6: Pre-allocate GPU Buffers Across NNI Iterations

**Files:**
- Modify: `scistreecna/scistreecna.py:357-500` (marginal_evaluate_dp_batch)
- Modify: `scistreecna/scistreecna.py:628-708` (nni_search_sinlge_round_batch)

**Why:** Every call to `marginal_evaluate_dp_batch` allocates ~4 large GPU arrays (all_U, all_U_, all__U, all_Q) of shape `(total_nodes, h, w)`. Since NNI search calls this repeatedly with similar-sized batches, we can pre-allocate once and reuse, just zeroing between calls.

- [ ] **Step 1: Add buffer cache to ScisTreeCNA class**

```python
def _get_eval_buffers(self, total_nodes, h, w):
    """Return pre-allocated GPU buffers, reallocating only if size changed."""
    needed = total_nodes * h * w * 4  # bytes per buffer
    key = (total_nodes, h, w)
    if not hasattr(self, '_buf_key') or self._buf_key != key:
        self._eval_buf = cp.zeros((4 * total_nodes, h, w), dtype=cp.float32)
        self._buf_key = key
    else:
        self._eval_buf[:] = 0  # zero out (faster than re-allocating)
    buf = self._eval_buf
    return (buf[0*total_nodes:1*total_nodes],
            buf[1*total_nodes:2*total_nodes],
            buf[2*total_nodes:3*total_nodes],
            buf[3*total_nodes:4*total_nodes])
```

- [ ] **Step 2: Use cached buffers in `marginal_evaluate_dp_batch`**

Replace:
```python
all_U  = cp.zeros((total_nodes, h, w), dtype=cp.float32)
all_U_ = cp.zeros((total_nodes, h, w), dtype=cp.float32)
all__U = cp.zeros((total_nodes, h, w), dtype=cp.float32)
# ... later ...
all_Q = cp.zeros((total_nodes, h, w), dtype=cp.float32)
```

with:
```python
all_U, all_U_, all__U, all_Q = self._get_eval_buffers(total_nodes, h, w)
```

And move the Q initialization (`all_Q[:, :, self.index_gt(2, 0)] = 1.0; all_Q = cp.log(all_Q)`) to happen after the buffer return. Note: `cp.log(all_Q)` creates a new array — we need to do it in-place:
```python
all_Q[:, :, self.index_gt(2, 0)] = 1.0
cp.log(all_Q, out=all_Q)  # in-place log, but log(0) = -inf which is what we want
```

Wait — `cp.log(cp.zeros(...))` gives `-inf` everywhere except the `index_gt(2,0)` column which is `log(1) = 0`. The buffer is already zeroed, so setting one column to 1.0 then taking log in-place works:
```python
# all_Q is already all zeros from _get_eval_buffers
all_Q[:, :, self.index_gt(2, 0)] = 1.0
# Now: all_Q has 0.0 everywhere except one column which is 1.0
# log(0) = -inf, log(1) = 0 — correct!
# But we need to be careful: cp.log(0) = -inf in float32
all_Q_log = cp.log(all_Q)  # keep as new array for safety
```

Actually, the simpler approach: just keep the allocation pattern but use `cp.empty` + explicit fill:
```python
all_Q = cp.full((total_nodes, h, w), float('-inf'), dtype=cp.float32)
all_Q[:, :, self.index_gt(2, 0)] = 0.0  # log(1) = 0
```

- [ ] **Step 3: Run benchmark**

```bash
python benchmark.py
```

- [ ] **Step 4: Commit**

```bash
git add scistreecna/scistreecna.py
git commit -m "perf: pre-allocate and reuse GPU eval buffers across NNI iterations"
```

---

### Task 7: Use CUDA Streams for Overlapped Kernel Execution

**Files:**
- Modify: `scistreecna/scistreecna.py:429-455` (Phase 3 up-pass)

**Why:** In the up-pass, leaf layers compute `U_ = matmul(U, tran_free)` and `_U = matmul(U, tran_mut)` — these two kernel launches are independent and can overlap on GPUs with sufficient SMs.

- [ ] **Step 1: Create two CUDA streams**

```python
stream_free = cp.cuda.Stream(non_blocking=True)
stream_mut = cp.cuda.Stream(non_blocking=True)
```

- [ ] **Step 2: Launch independent kernels on separate streams**

For leaf layers:
```python
with stream_free:
    self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation_free, ptr_u_, nb, h, w, shared_mem)
with stream_mut:
    self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation, ptr__u, nb, h, w, shared_mem)
stream_free.synchronize()
stream_mut.synchronize()
```

For internal layers, the `_U` computation (mutation matmul) is independent of the `U_` computation (fused add+mutation_free matmul), so they can also overlap:
```python
with stream_free:
    self._launch_batched_add_matmul(...)
    self._launch_batched_matadd(...)
with stream_mut:
    self._launch_batched_matmul(_matmul, ptr_u, self.tran_prob_mutation, ptr__u, ...)
```

Wait — for internal layers, `ptr_u` depends on `_launch_batched_matadd` completing. So the mutation matmul must wait. Only the leaf case can overlap cleanly.

- [ ] **Step 3: Run benchmark and check if streams help**

```bash
python benchmark.py
```

Note: stream overlap benefit depends on GPU occupancy. If the GPU is already fully utilized by single kernels, streams won't help. Profile with `nsys` or CuPy profiler to verify.

- [ ] **Step 4: Commit if beneficial, revert if not**

```bash
git add scistreecna/scistreecna.py
git commit -m "perf: use CUDA streams for overlapped kernel execution in up-pass"
```

---

### Task 8: End-to-End Validation and Profiling

**Files:**
- No code changes
- Run: `benchmark.py`, `profile_detailed.py`

- [ ] **Step 1: Run full benchmark**

```bash
python benchmark.py
```

Record: total time, tree accuracy, genotype accuracy, likelihood.
Compare against baseline (3.5s for 10 iterations).

- [ ] **Step 2: Run detailed profiler**

```bash
python profile_detailed.py
```

Record per-stage timing breakdown. Identify if further optimization is warranted.

- [ ] **Step 3: Profile with CUDA profiler if available**

```bash
nsys profile -o profile_report python benchmark.py
# or
python -c "
import cupy as cp
cp.cuda.profiler.start()
# ... run one NNI iteration ...
cp.cuda.profiler.stop()
"
```

- [ ] **Step 4: Document results**

Update `OPTIMIZATION_REPORT.md` with Phase 2 results table:
```
| Version | Time (10 iterations) | Speedup |
|---------|---------------------|---------|
| v3 (Phase 1 final) | ~3.5s | 3.9x |
| v4 (Phase 2) | ~X.Xs | Y.Yx |
```

- [ ] **Step 5: Commit**

```bash
git add OPTIMIZATION_REPORT.md
git commit -m "docs: update optimization report with Phase 2 results"
```

---

## Priority Order

**High confidence, high impact:**
1. **Task 5** — Remove redundant struct_copy_tree (free ~27ms/round, zero risk)
2. **Task 1** — Cache transpose (trivial, eliminates repeated allocation)
3. **Task 2** — Single contiguous allocation (reduces allocator overhead)

**Medium confidence, medium impact:**
4. **Task 6** — Pre-allocate buffers across iterations (saves allocation cost)
5. **Task 3** — Optimize 3vecdot kernel (reduces global memory reads)
6. **Task 4** — Fix O(n^2) topological sort (scales better with large batches)

**Experimental (profile first):**
7. **Task 7** — CUDA streams (may not help if GPU already saturated)

**Always do last:**
8. **Task 8** — End-to-end validation and profiling
