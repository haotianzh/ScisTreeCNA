# Command-line interface

Installing ScisTreeCNA adds the `scistreecna` command to your `PATH`. The command reads a CSV file, runs the same inference procedure as the Python API, and writes the inferred tree and genotype matrix to disk. It is suitable for individual analyses and batch jobs. Use the [Python API](tutorials/usage.ipynb) if you need the tree object or copy-number event mappings.

## Basic usage

```bash
scistreecna --input ./examples/test_data_reads.csv --output test_output
```

This command produces two files:

| File | Contents |
| :--- | :--- |
| `test_output_tree.txt` | Inferred cell lineage tree in Newick format |
| `test_output_genotype.txt` | Called binary genotype matrix as tab-separated integers, with sites as rows and cells as columns |

The CLI accepts both total copy-number input (`ref|alt|cn`) and allele-specific copy-number input (`ref|alt|cn_maj|cn_min`). It detects the format automatically. See the [input-format guide](tutorials/input.ipynb) for details.

## Options

```text
usage: scistreecna [-h] --input INPUT [--output OUTPUT] [--cn-min CN_MIN]
                   [--cn-max CN_MAX] [--ado ADO] [--seq-error SEQ_ERROR]
                   [--af AF] [--max-iter MAX_ITER] [--cn-noise CN_NOISE]
                   [--tree-batch TREE_BATCH] [--node-batch NODE_BATCH]
                   [--verbose]
```

### Input and output

`--input`, `-i`
: Path to the input CSV file. Required.

`--output`, `-o`
: Prefix used for the output files: `{prefix}_tree.txt` and `{prefix}_genotype.txt`. Default: `output`.

### Model parameters

`--cn-min`
: Minimum copy number allowed in the generalized-genotype state space. Must be greater than `0`. Default: `1`.

`--cn-max`
: Maximum copy number allowed in the state space. Default: `5`. Together, `--cn-min` and `--cn-max` determine the number of possible states and the computational cost. See the [performance guide](performance.md).

`--ado`
: Allelic dropout rate. Default: `0.1`.

`--seq-error`
: Sequencing error rate. Default: `0.01`.

`--cn-noise`
: Copy-number noise. Default: `0.05`. For clone-averaged copy numbers, consider increasing this value to approximately `0.5`.

`--af`
: Expected mutant-allele frequency at a heterozygous site. Default: `0.5`.

### Search and performance

`--max-iter`
: Maximum number of NNI local-search iterations. A value of `0` runs the search until convergence. Default: `0`.

`--tree-batch`
: Number of candidate trees evaluated together on the GPU. Larger values can improve speed but require more GPU memory. Default: `64`.

`--node-batch`
: Retained for API compatibility; it has no effect in the current implementation. Default: `64`.

`--verbose`
: Print local-search progress. Disabled by default.

:::{note}
The default `--tree-batch` value of `64` is conservative. Larger GPUs may support higher values and faster inference. Use `estimate_batch_sizes` to select a value that fits in memory, as described in the [performance guide](performance.md).
:::

## Errors and exit status

The command exits with a nonzero status if the input file is missing or invalid, or if inference fails. The error is printed to the console without a stack trace. To obtain a full stack trace for debugging, run `scistreecna.infer` through the Python API.