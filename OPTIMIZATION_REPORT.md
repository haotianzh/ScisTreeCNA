# ScisTreeCNA GPU Optimization Report

## Performance Summary

| Version | Time (10 iterations) | Speedup |
|---------|---------------------|---------|
| **Baseline** (original) | ~13.8s | 1.0x |
| **v1**: kernel opts + struct_copy | ~8.0s | 1.7x |
| **v2**: + contiguous GPU arrays | ~5.1s | 2.7x |
| **v3**: + base tree in batch | **~3.5s** | **3.9x** |

Results are numerically identical across all versions:
- Likelihood: -12423.075
- Tree accuracy: 0.4828
- Genotype accuracy: 0.9908

---

## 1. GPU Kernel Optimizations (`scistreecna/util/kernels.py`)

### 1.1 `kernel_log_matmul` — Shared Memory + Two-Pass Log-Sum-Exp

**Before:**
```c
// mat1: (nsite, k) mat2: (k, k)
extern "C" __global__ void log_matmul(float* mat1, float* mat2, float* out, int n, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < k) {
        out[i*k+j] = logf(0.0f);
        for (int p=0; p<k; p++){
            float m = fmaxf(out[i*k+j], mat1[i*k+p] + mat2[j*k+p]);
            if (!__isinf(m))
                out[i*k+j] = m + logf(expf(out[i*k+j] - m) + expf(mat1[i*k+p] + mat2[j*k+p] - m));
        }
    }
}
```

**After:**
```c
// Optimized: shared memory for mat2 + two-pass log-sum-exp
extern "C" __global__ void log_matmul(float* mat1, float* mat2, float* out, int n, int k){
    extern __shared__ float s_mat2[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_local = threadIdx.x * blockDim.y + threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;

    // Cooperatively load mat2 into shared memory
    for (int idx = tid_local; idx < k * k; idx += total_threads) {
        s_mat2[idx] = mat2[idx];
    }
    __syncthreads();

    if (i < n && j < k) {
        // Two-pass log-sum-exp (numerically stable)
        float maxval = -1.0f / 0.0f;
        for (int p = 0; p < k; ++p) {
            float val = mat1[i*k + p] + s_mat2[j*k + p];
            if (val > maxval) maxval = val;
        }
        float sumexp = 0.0f;
        if (!isinf(maxval) || maxval > 0.0f) {
            for (int p = 0; p < k; ++p) {
                float val = mat1[i*k + p] + s_mat2[j*k + p];
                if (!isinf(val)) sumexp += expf(val - maxval);
            }
        }
        out[i*k + j] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
    }
}
```

**Key changes:**
- `mat2` (transition matrix, k×k) loaded into shared memory once per block — all threads reuse it
- Two-pass log-sum-exp: first find max, then accumulate — avoids writing intermediate results to global memory
- Same change applied to `kernel_batch_log_matmul`

### 1.2 `kernel_batch_log_matmul` — Same Optimizations + Bug Fix

**Before:**
```c
extern "C" __global__ void batch_log_matmul(float** bmat1, float* mat2, float** bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z > m)           // BUG: off-by-one, should be z >= m
        return;
    // ... sequential online log-sum-exp reading/writing global memory ...
}
```

**After:**
```c
extern "C" __global__ void batch_log_matmul(float** bmat1, float* mat2, float** bout, int m, int n, int k){
    extern __shared__ float s_mat2[];

    int z = blockIdx.z;
    if (z >= m) return;   // FIXED: off-by-one

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_local = threadIdx.x * blockDim.y + threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;

    // Cooperatively load mat2 (k x k) into shared memory
    for (int idx = tid_local; idx < k * k; idx += total_threads) {
        s_mat2[idx] = mat2[idx];
    }
    __syncthreads();

    if (i < n && j < k) {
        float* mat1 = bmat1[z];
        float* out = bout[z];

        // Two-pass log-sum-exp
        float maxval = -1.0f / 0.0f;
        for (int p = 0; p < k; ++p) {
            float val = mat1[i*k + p] + s_mat2[j*k + p];
            if (val > maxval) maxval = val;
        }
        float sumexp = 0.0f;
        if (!isinf(maxval) || maxval > 0.0f) {
            for (int p = 0; p < k; ++p) {
                float val = mat1[i*k + p] + s_mat2[j*k + p];
                if (!isinf(val)) sumexp += expf(val - maxval);
            }
        }
        out[i*k + j] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
    }
}
```

### 1.3 Off-by-one Bug Fixes in `batch_matadd` and `batch_matadd_stride`

**Before:** `if (z > m) return;`
**After:** `if (z >= m) return;`

### 1.4 New Fused Kernel: `kernel_batch_add_log_matmul`

Fuses `batch_matadd` + `batch_log_matmul` into one kernel — saves one kernel launch and one global memory round-trip per call.

```c
// Fused kernel: batch_matadd + batch_log_matmul in one launch
// Computes: out[z] = log_matmul(mat_a[z] + mat_b[z], mat2)
extern "C" __global__ void batch_add_log_matmul(
    float** bmat_a, float** bmat_b, float* mat2, float** bout,
    int m, int n, int k)
{
    extern __shared__ float s_mat2[];

    int z = blockIdx.z;
    if (z >= m) return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_local = threadIdx.x * blockDim.y + threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;

    for (int idx = tid_local; idx < k * k; idx += total_threads) {
        s_mat2[idx] = mat2[idx];
    }
    __syncthreads();

    if (i < n && j < k) {
        float* mat_a = bmat_a[z];
        float* mat_b = bmat_b[z];
        float* out = bout[z];

        float maxval = -1.0f / 0.0f;
        for (int p = 0; p < k; ++p) {
            float val = (mat_a[i*k + p] + mat_b[i*k + p]) + s_mat2[j*k + p];
            if (val > maxval) maxval = val;
        }
        float sumexp = 0.0f;
        if (!isinf(maxval) || maxval > 0.0f) {
            for (int p = 0; p < k; ++p) {
                float val = (mat_a[i*k + p] + mat_b[i*k + p]) + s_mat2[j*k + p];
                if (!isinf(val)) sumexp += expf(val - maxval);
            }
        }
        out[i*k + j] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
    }
}
```

**Wrapper function:**
```python
def batch_add_log_matmul_cuda():
    return cp.RawKernel(kernel_batch_add_log_matmul, "batch_add_log_matmul")
```

### 1.5 Kernel-Level Benchmark

| Kernel | Before | After | Speedup |
|--------|--------|-------|---------|
| `batch_log_matmul` (32,32) block | 0.362 ms | — | — |
| `batch_log_matmul` (8,32) + shmem | — | 0.015 ms | **24x** |
| `fused add+log_matmul` (8,32) + shmem | — | 0.019 ms | (new) |

---

## 2. Launch Parameter Optimizations (`scistreecna/scistreecna.py`)

### 2.1 Dynamic Block Size

**Before:** Hard-coded `block_size = (32, 32)` — with k=20 (genotype states), 12 of 32 y-threads idle (37.5% waste).

**After:** Dynamic computation matching actual dimension:
```python
def _get_block_grid(self, h, w, batch):
    block_y = min(32, w)
    block_x = max(1, min(256 // block_y, 32))
    block_size = (block_x, block_y)
    grid_size = (
        (h + block_size[0] - 1) // block_size[0],
        (w + block_size[1] - 1) // block_size[1],
        batch,
    )
    return block_size, grid_size
```

For w=20: `block_size = (12, 20)` — 100% thread utilization.

### 2.2 Shared Memory Parameter

All kernel calls now pass `shared_mem=w*w*4`:
```python
util.batch_log_matmul_cuda()(
    grid_size, block_size,
    (ptr_u, self.tran_prob_mutation_free, ptr_u_, nb, h, w),
    shared_mem=shared_mem)
```

---

## 3. Contiguous GPU Array + Index-Based Pointers (`scistreecna/scistreecna.py`)

This is the most impactful algorithmic optimization. The old `marginal_evaluate_dp_batch` spent 99% of time in Python loops (assigning arrays to node objects, building pointer arrays via list comprehensions). The new version eliminates all per-node Python work.

### 3.1 Old Approach (Python-loop bottleneck)

```python
# Per-layer Python loop: ~6ms per layer × 25 layers = 150ms
for node in nodes:
    node.U = zeros[cur]
    node._U = zeros[cur + 1]
    node.U_ = zeros[cur + 2]
    cur += 3
    if node.is_leaf():
        node.U = probs[int(node.name)]

# Pointer construction via list comprehension: ~3ms per call
ptr_u = cp.array([n.U.data.ptr for n in nodes])
ptr_c0_u_ = cp.array([n.get_children()[0].U_.data.ptr for n in nodes])
```

### 3.2 New Approach (GPU arithmetic)

**Phase 1: One-time topology pre-computation** (~10ms total for 128 trees):
```python
# Assign flat index to each node
node_id_map = {}  # id(node) -> flat index
idx = 0
for tid, tree in enumerate(trees):
    for nid, node in tree._nodes.items():
        node_id_map[id(node)] = idx
        idx += 1

# Build per-layer GPU index arrays
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
```

**Phase 2: Contiguous GPU arrays** (one allocation, no per-node assignment):
```python
all_U  = cp.zeros((total_nodes, h, w), dtype=cp.float32)
all_U_ = cp.zeros((total_nodes, h, w), dtype=cp.float32)
all__U = cp.zeros((total_nodes, h, w), dtype=cp.float32)
base_U  = all_U.data.ptr   # single int64 base address
base_U_ = all_U_.data.ptr
base__U = all__U.data.ptr
stride = h * w * 4          # bytes per (h, w) matrix
```

**Phase 3: GPU arithmetic pointer construction** (~0.01ms vs ~3ms):
```python
# OLD: Python list comprehension + CuPy array creation
ptr_u = cp.array([n.U.data.ptr for n in nodes])         # ~3ms

# NEW: Pure GPU integer arithmetic
ptr_u = base_U + indices * stride                        # ~0.01ms
ptr_c0_u_ = base_U_ + c0_indices * stride               # ~0.01ms
```

Both produce the same result: a CuPy int64 array of memory addresses, which the kernel interprets as `float**`.

**Phase 3-5: Computation uses the same kernels**, just with GPU-computed pointer arrays:
```python
for layer_info in up_layers:
    if layer_info[0] == 'leaf':
        _, indices, cell_ids = layer_info
        all_U[indices] = probs[cell_ids]  # CuPy fancy indexing (GPU)
        ptr_u  = base_U  + indices * stride
        ptr_u_ = base_U_ + indices * stride
        # ... launch kernels with ptr_u, ptr_u_ ...
    else:
        _, indices, c0, c1 = layer_info
        ptr_c0_u_ = base_U_ + c0 * stride
        ptr_c1_u_ = base_U_ + c1 * stride
        ptr_u  = base_U  + indices * stride
        # ... launch kernels ...
```

### 3.3 Time Breakdown Comparison

| Stage | Before | After |
|-------|--------|-------|
| Python node loop (assign U/U_/_U) | 145 ms | **0 ms** (eliminated) |
| Pointer array construction | 20 ms | **~0.3 ms** (GPU arithmetic) |
| GPU kernels | 4 ms | 4 ms |
| **Total eval 128 trees** | **431 ms** | **61 ms** (7x) |

---

## 4. Base Tree In-Batch Evaluation (`scistreecna/scistreecna.py`)

### 4.1 Old Approach

```python
# Separate single-tree eval (91ms) using old per-node Python loop path
best_likelihood, indicies = self.marginal_evaluate_dp(probs, struct_copy_tree(best_tree))
# Then batch eval candidates
for bi, trees in enumerate(loader()):
    likelihoods = self.marginal_evaluate_dp_batch(probs, ...)
```

### 4.2 New Approach

```python
# Include base tree as index 0 in the candidate batch
all_trees = [best_tree] + candidates
loader = TreeBatchLoader(all_trees, batch_size=tree_batch_size)
best_likelihood = cp.float64(-np.inf)
for bi, trees in enumerate(loader()):
    likelihoods = self.marginal_evaluate_dp_batch(probs, ...)
    # base tree competes with candidates in the same batch
```

Saves ~91ms per NNI iteration by eliminating the separate single-tree evaluation.

---

## 5. Lightweight Tree Copy (`scistreecna/base/tree.py`)

### 5.1 Old Approach

```python
def copy(self):
    return cPickle.loads(cPickle.dumps(self, -1))  # 3.34 ms/call
```

Pickle-based deepcopy serializes the entire tree including any CuPy arrays attached to nodes.

### 5.2 New Approach

```python
def struct_copy_tree(tree):
    """Standalone lightweight tree copy. Only copies structure, not GPU arrays."""
    old_to_new = {}
    nodes = tree._nodes if hasattr(tree, '_nodes') else tree.get_all_nodes()
    node_items = nodes.items() if isinstance(nodes, dict) else ...

    # First pass: create new Node objects (no CuPy arrays)
    for nid, old_node in node_items:
        new_node = Node.__new__(Node)
        new_node._identifier = old_node._identifier
        new_node.name = old_node.name
        new_node._branch = old_node._branch
        new_node.parent = None
        new_node.children = OrderedDict()
        new_node._children = []
        old_to_new[nid] = new_node

    # Second pass: rebuild parent-child links
    for nid, old_node in node_items:
        new_node = old_to_new[nid]
        if old_node.parent is not None:
            new_node.parent = old_to_new[old_node.parent.identifier]
        for child in old_node._children:
            new_child = old_to_new[child.identifier]
            new_node.children[child.identifier] = new_child
            new_node._children.append(new_child)

    new_tree = BaseTree.__new__(BaseTree)
    new_tree.root = old_to_new[tree.root.identifier]
    new_tree._nodes = old_to_new
    return new_tree
```

**Performance:** 0.21 ms/call vs 3.34 ms/call = **15.6x faster**.

Works with both `scistreecna.base.tree.BaseTree` and `scistree2.tree.BaseTree`.

---

## Files Changed

1. **`scistreecna/util/kernels.py`** — Optimized CUDA kernels + new fused kernel
2. **`scistreecna/scistreecna.py`** — Contiguous GPU arrays, index-based pointers, dynamic block sizes, base tree in batch
3. **`scistreecna/base/tree.py`** — Lightweight `struct_copy_tree()` function
