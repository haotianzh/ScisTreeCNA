# Performance and tuning

Most of ScisTreeCNA’s runtime is spent scoring candidate trees during NNI local search. Each tree is evaluated across all sites and generalized-genotype states, and a search may examine thousands of candidates. ScisTreeCNA accelerates this process by evaluating batches of candidate trees on the GPU.

The main performance considerations are therefore the number of genotype states and the number of trees evaluated per batch.

## What determines computational cost

Peak GPU memory is dominated by four `float32` arrays with shape

$$
\left(
\texttt{tree\_batch\_size}(2n_{\text{cells}}-1),
n_{\text{sites}},
N
\right),
$$

plus a leaf-likelihood array with shape

$$
(n_{\text{cells}}, n_{\text{sites}}, N).
$$

Memory usage therefore grows approximately linearly with the numbers of cells, sites, generalized-genotype states, and trees per batch.

The number of states, $N$, is determined by the modeled copy-number range:

$$
N =
\frac{
(\texttt{cn\_max}-\texttt{cn\_min}+1)
(\texttt{cn\_max}+\texttt{cn\_min}+2)
}{2}.
$$

| `cn_min` | `cn_max` | $N$ |
| ---: | ---: | ---: |
| 1 | 3 | 9 |
| 1 | 5 | 20 |
| 1 | 8 | 40 |
| 1 | 10 | 60 |

Increasing `cn_max` from `5` to `10` triples the number of states and approximately triples both memory use and computation. Set the copy-number range to values supported by the data instead of extending it unnecessarily.

## Choose a batch size

Use `estimate_batch_sizes` to select a `tree_batch_size` based on the available GPU memory:

```python
import scistreecna as scna

info = scna.estimate_batch_sizes(
    n_cells=500,
    n_sites=2000,
    cn_min=1,
    cn_max=5,
    gpu_id=0,
    mem_fraction=0.75,
)

print(info["tree_batch_size"])
print(info["estimated_mem_usage_MB"])

tree, geno = scna.infer(
    reads,
    cell_names=cell_names,
    tree_batch_size=info["tree_batch_size"],
)
```

Here, `mem_fraction` specifies the fraction of currently free GPU memory that may be used. The returned dictionary also reports:

- `N_states`
- `nodes_per_tree`
- `bytes_per_tree_MB`
- `gpu_free_MB`
- `gpu_total_MB`

These values can help identify the main memory cost when a dataset does not fit.

The estimated batch size is also limited by:

- The number of available NNI candidates, approximately $4(n_{\text{cells}}-1)$.
- CUDA’s maximum grid size, which limits the batch to approximately $\lfloor 65535/n_{\text{cells}}\rfloor$.

A batch size smaller than expected may therefore reflect one of these limits rather than insufficient GPU memory.

:::{tip}
The default `tree_batch_size` is `64` in both the Python API and CLI. This conservative value is suitable for small GPUs, but larger GPUs may support much larger batches. Use `estimate_batch_sizes` to improve throughput without exceeding the available memory.
:::

`node_batch_size` remains available for compatibility with existing scripts but has no effect on the current optimized evaluation procedure.

## When a run is too large or slow

Try the following, in order:

1. **Reduce `tree_batch_size`.** This lowers memory use without changing the result, but may increase runtime.
2. **Narrow the copy-number range.** Reducing `cn_max` can greatly reduce the number of states. This may change the result because states outside the specified range cannot be inferred.
3. **Reduce the number of sites.** Restrict the analysis to reliable variant sites to reduce memory use and computation.
4. **Set `max_iter`.** Limiting the number of search iterations reduces runtime but may stop the search before convergence. It does not reduce peak memory use.
5. **Free GPU memory or use a larger GPU.** The batch-size estimate is based on currently free memory, so other processes using the GPU will reduce the recommended batch size.

## Choose model parameters

These parameters primarily affect inference accuracy rather than speed.

`ado` (default: `0.1`)
: Allelic dropout rate. Set this according to the sequencing protocol; whole-genome amplification may produce substantially higher dropout rates.

`seq_error` (default: `0.01`)
: Per-base sequencing error rate.

`cn_noise` (default: `0.05`)
: Uncertainty in the observed copy numbers. The default is suitable for cell-specific copy-number calls. For clone-averaged copy numbers, consider increasing this value to approximately `0.5`.

`cn_min` and `cn_max` (defaults: `1` and `5`)
: Modeled copy-number range. `cn_min` must be greater than `0`. An observed copy number of `0` is allowed in the input, although total copy number `0` is not included in the latent state space.

`af` (default: `0.5`)
: Expected mutant-allele fraction at a heterozygous site.

## Monitor a run

Set `verbose=True` to display the likelihood after each local-search iteration:

```python
tree, geno = scna.infer(
    reads,
    cell_names=cell_names,
    verbose=True,
)
```

If the likelihood is still increasing when the run reaches `max_iter`, increase the iteration limit or allow the search to run until convergence.