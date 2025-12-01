<p align="center"><img width=300; src="https://raw.githubusercontent.com/haotianzh/ScisTreeCNA/refs/heads/main/imgs/logo.png"/></p>
<p align="center">   
  <a href="https://colab.research.google.com/drive/1roB2pnTBlFvoQtCNn0QDP3NgvGtK97Yl?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"/></a>&thinsp;<a href="https://github.com/haotianzh/ScisTreeCNA/issues"><img src="https://img.shields.io/github/issues/haotianzh/ScisTreeCNA" alt="GitHub issues"/></a>&thinsp;<a href=""><img src="https://img.shields.io/github/license/yufengwudcs/ScisTree2?color=%239b02fa"/></a>&thinsp;<a href="https://www.biorxiv.org/content/10.1101/2025.11.21.689819v1"><img alt="biorxiv" src="https://img.shields.io/badge/10.1101%2F2025.11.21.689819-red?label=DOI&color=%23ff0000&link=https%3A%2F%2Fwww.biorxiv.org%2Fcontent%2F10.1101%2F2025.11.21.689819v1"></a>
</p>
<!-- &thinsp;<a href="https://github.com/username/repo"><img src="https://img.shields.io/github/stars/haotianzh/ScisTreeCNA?style=social" alt="GitHub stars"/></a> -->
<!-- **Accurate Probabilistic Reconstruction of Cell Lineage Trees from SNVs and CNAs** -->
*This repository contains the code for the paper **"Accurate Probabilistic Reconstruction of Cell Lineage Trees from SNVs and CNAs with ScisTreeCNA"**, submitted for publication, 2025. Here is the [preprint](https://www.biorxiv.org/content/10.1101/2025.11.21.689819v1).*

ScisTreeCNA is a **GPU-accelerated** Python package designed for reconstructing **Cell Lineage Trees (CLTs)** from single-cell DNA sequencing data. It addresses a key challenge by integrating both single nucleotide variants (**SNVs**) and copy number abberations (**CNAs**) within a unified **probabilistic** framework. This approach provides accurate and scalable inference for modern single-cell datasets containing both SNV and CNA information.

> **Note:** If you do not have copy-number data and want to infer a cell lineage tree from SNVs only, please use [ScisTree2](https://github.com/yufengwudcs/ScisTree2).

---

## ⚠️ Important Note: GPU Requirement

ScisTreeCNA relies exclusively on **CUDA** to achieve high-performance probabilistic reconstruction. CPU-only environments are **not** supported.

However, anyone can try ScisTreeCNA using Google Colab with free T4 GPU access:  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1roB2pnTBlFvoQtCNn0QDP3NgvGtK97Yl?usp=sharing).  To get started with colab, first make your own copy of the notebook and then you can modify your copy to try to use new data.


## ⚙️ System Requirements

| Component | Requirement | Note |
| :--- | :--- | :--- |
| **Operating System** | Linux/macOS/Windows | |
| **GPU** | NVIDIA CUDA GPU | Compute Capability 3.0 or higher is required. |
| **Toolkit** | CUDA Runtime | Essential for GPU operation and can typically be installed alongside CuPy (see [Installation](#-installation)). |
||Python & Pip | Python $\ge$ 3.8 |
||Conda | Miniconda/Anaconda |

**We have successfully tested it on Linux, macOS, and Windows.*
***


## 🚀 Installation

### Option 1: Recommended Installation (Using Conda to Manage CUDA)

This method simplifies dependency management by allowing `conda` to install the matched CuPy and CUDA Runtime Libraries together.

0. You can either create a fresh environment or use an existing one.
    ```bash
    conda create -n scistreecna python=3.12
    conda activate scistreecna
    ```
1.  **Install CuPy along with CUDA Runtime**
    ```bash
    # Example: Install CuPy with CUDA 12.8 Runtime
    conda install -c conda-forge cupy cuda-version=12.8
    ```

    > **Note:** Installing CuPy with `conda` automatically manages and installs the specific **CUDA runtime libraries** required for ScisTreeCNA to operate, greatly simplifying the setup. Check for more details at the [CuPy official website](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge).

2.  **Install ScisTreeCNA**

    Clone the repository and install the package using `pip`. Since CUDA dependencies are already handled by CuPy, use the standard local installation:

    ```bash
    git clone https://github.com/haotianzh/ScisTreeCNA.git
    cd ScisTreeCNA
    pip install .
    ```

---

### Option 2: Installation with Pre-Installed CUDA Toolkit

If you already have the **NVIDIA CUDA Toolkit** installed on your system (or your current conda env) and only need to install ScisTreeCNA and the CuPy that version compatible with your existing setup, you can try:

1.  **Clone ScisTreeCNA**
    ```bash
    git clone https://github.com/haotianzh/ScisTreeCNA.git
    cd ScisTreeCNA
    ```

2.  **Install ScisTreeCNA with Specific CuPy Dependency**

    Use the `pip install .[extra]` syntax, replacing the extra name with your CUDA major version to ensure compatibility:

    * For CUDA 11.x:
        ```bash 
        pip install .[cuda11x] 
        ```
    * For CUDA 12.x:
        ```bash 
        pip install .[cuda12x] 
        ```
    * For CUDA 13.x:
        ```bash 
        pip install .[cuda13x] 
        ```


## 💡 Usage
Once installed, ScisTreeCNA can be used directly from your Python environment.
### Input
To run the inference, you must prepare an input `.csv` file where **rows represent sites** and **columns represent cells**. For each *(cell, site)* pair, the entry is a string in the format **`ref|alt|cn`**, where:
- **ref**: read count of the reference (wild-type) allele  
- **alt**: read count of the mutant allele  
- **cn**: observed copy number (either absolute copy number — recommended — or relative copy state)

Missing values should be encoded as:

- `.|.|cn` — if read counts are missing but copy number is available  
- `ref|alt|.` - if only copy number is missing
- `.|.|.` — if both read counts and copy number are missing

 Example input files are provided in the `examples/` directory for reference.

### Python Interface

This example demonstrates loading example data and running the tree reconstruction function.

```python
import scistreecna as scna
# scna.set_cuda_device(1) # set to gpu:1
# load example data
reads, cell_names, site_names = scna.util.read_csv('./examples/test_data_reads.csv')
# run inference
scistreecna_tree, scistreecna_geno = scna.infer(
    reads,
    cell_names=cell_names,  # cell names
    ado=0.1,  # allelic dropout rate
    seq_error=0.01,  # sequencing error
    cn_noise=0.05,  # copy number noise
    cn_min=1,  # minimum copy number (>=1)
    cn_max=5,  # maximum copy number
    tree_batch_size=128,  # number of trees evaluated in parallel
    node_batch_size=256,  # number of nodes evaluated in parallel
    verbose=True, # print logs
)  
print(scistreecna_tree)  # print inferred tree
print(scistreecna_geno)  # print imputed binary genotype
```
More usage examples can be found in our [tutorials](tutorials/scistreecna_basic.ipynb).



### Command-Line Interface

We also provide a CLI tool that accepts a `.csv` file as input and saves the inferred tree and imputed genotypes to a user-specified output location.

To run inference on the example data using the CLI:

```bash
scistreecna --input ./examples/test_data_reads.csv --output test_output
```
Detailed usage:
```
usage: scistreecna [-h] --input INPUT [--output OUTPUT] [--cn-min CN_MIN] [--cn-max CN_MAX] [--ado ADO] [--seq-error SEQ_ERROR] [--af AF] [--max-iter MAX_ITER] [--cn-noise CN_NOISE] [--tree-batch TREE_BATCH] [--node-batch NODE_BATCH] [--verbose]

CLI for ScisTreeCNA inference.

options:
  -h, --help                    show this help message and exit
  --input INPUT, -i INPUT       Path to input reads file (see https://github.com/haotianzh/ScisTreeCNA/blob/main/examples/test_data_reads.csv)
  --output OUTPUT, -o OUTPUT    Prefix for output files. Saves as {prefix}_tree.txt and {prefix}_genotype.txt (default: 'output')
  --cn-min CN_MIN               Minimum copy number (default: 1)
  --cn-max CN_MAX               Maximum copy number (default: 5)
  --ado ADO                     Allelic dropout rate (default: 0.1)
  --seq-error SEQ_ERROR         Sequencing error rate (default: 0.01)
  --af AF                       Allele Frequency (default: 0.5)
  --max-iter MAX_ITER           Maximal local search iteration (default: infinity)
  --cn-noise CN_NOISE           Copy number noise (default: 0.05)
  --tree-batch TREE_BATCH       Tree batch size (default: 64)
  --node-batch NODE_BATCH       Node batch size (default: 64)
  --verbose                     Enable verbose logs (default: False)
```


## Simulator
The accompanying simulator, **scsim**, for generating reads with copy-number gains and losses is available [here](https://github.com/haotianzh/scsim).


## Contact
Post your issues here inside GitHub repositary if you have questions/issues.
