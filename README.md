<h1 align="center"><img width=300; src="imgs/logo.png"></h1>

<p align="center">   
  <a href="https://colab.research.google.com/drive/1roB2pnTBlFvoQtCNn0QDP3NgvGtK97Yl?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="GitHub stars"/></a>&thinsp;<a href="https://github.com/haotianzh/ScisTreeCNA/issues"><img src="https://img.shields.io/github/issues/haotianzh/ScisTreeCNA" alt="GitHub issues"/></a>&thinsp;<a href=""><img src="https://img.shields.io/github/license/yufengwudcs/ScisTree2?color=%239b02fa"/>&thinsp;<a href="https://github.com/username/repo"><img src="https://img.shields.io/github/stars/haotianzh/ScisTreeCNA?style=social" alt="GitHub stars"/></a>
</p>

<!-- **Accurate Probabilistic Reconstruction of Cell Lineage Trees from SNVs and CNAs** -->
This repository contains the code for the paper **"Accurate Probabilistic Reconstruction of Cell Lineage Trees from SNVs and CNAs with ScisTreeCNA"** submitted to **RECOMB 2026** for review.

ScisTreeCNA is a **GPU-accelerated** Python package designed for reconstructing **Cell Lineage Trees (CLTs)** from single-cell DNA sequencing data. It addresses a key challenge by integrating both Single-Nucleotide Variants (**SNVs**) and Copy-Number Alterations (**CNAs**) within a unified **probabilistic** framework. This approach provides accurate and scalable inference, making it ideal for modern single-cell datasets containing both SNV and CNA information.

***

## ⚠️ Important Note: GPU Requirement

ScisTreeCNA relies exclusively on **NVIDIA CUDA GPUs** to achieve its high-performance, probabilistic reconstruction.

**CPU-only environments are not supported.**

***

## ⚙️ System Requirements

| Component | Requirement | Note |
| :--- | :--- | :--- |
| **Operating System** | Linux / Windows | |
| **GPU** | NVIDIA CUDA GPU | Compute Capability **3.0** or higher is required. |
| **Toolkit** | CUDA Runtime | Essential for GPU operation and can typically be installed alongside CuPy (see Installation). |
***


## 🚀 Installation

### Option 1: Recommended Installation (Using Conda to Manage CUDA)

This method simplifies dependency management by allowing `conda` to install the matched CuPy and CUDA Runtime Libraries together.

0. Create a fresh environment
    ```bash
    conda create -n scistreecna python=3.12
    conda activate scistreecna
    ```
1.  **Install CuPy and CUDA Runtime**
    ```bash
    # Example: Install CuPy with CUDA 12.x Runtime
    conda install -c conda-forge cupy 
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

If you already have the **NVIDIA CUDA Toolkit** installed on your system and only need to install ScisTreeCNA and the CuPy version compatible with your existing setup, you can try:

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

Once installed, you can use ScisTreeCNA from your Python environment. You will need to prepare your input files, see examples in folder `examples/`.

### Python Interface

This example demonstrates loading placeholder data paths and running the primary tree reconstruction function.

```python
import scistreecna as scna
# load example data
reads, cell_names, site_names = scna.util.read_csv('./examples/test_data_reads.csv')
# run inference
scistreecna_tree, scistreecna_geno = scna.infer(reads,
                                                cell_names=cell_names,  # cell names
                                                ado=0.1,  # allelic dropout rate
                                                seq_error=0.01,   # sequencing error
                                                cn_noise=0.05,    # copy number noise
                                                cn_min=1, # minimum copy number (>=1) 
                                                cn_max=5, # maximum copy number
                                                tree_batch_size=128,  # number of trees evaluated together
                                                node_batch_size=256,  # number of nodes evaluated together 
                                                verbose=True)  # print logs
print(scistreecna_tree) # print inferred tree
print(scistreecna_geno) # print imputed binary genotype
```
More usage examples can be found in our [tutorials](tutorials/scistreecna_basic.ipynb).



### Command-Line Interface

We also provide a CLI tool that accepts a `.csv` file as input and saves the inferred tree and imputed genotypes to a user-specified output location.

To run inference on the example data using the CLI:

```bash
scistreecna --input ./examples/test_data_reads.csv --o test_output
```
Detailed usage:
```
usage: scistreecna [-h] --input INPUT [--output OUTPUT] [--cn-min CN_MIN] [--cn-max CN_MAX] [--ado ADO] [--seq-error SEQ_ERROR] [--af AF] [--cn-noise CN_NOISE] [--tree-batch TREE_BATCH] [--node-batch NODE_BATCH] [--verbose]

CLI for ScisTreeCNA inference.

options:
  -h, --help            Show this help message and exit
  --input INPUT, -i INPUT
                        Path to input reads file (.npy format expected for 3D arrays)
  --output OUTPUT, -o OUTPUT
                        Prefix for output files. Saves as {prefix}_tree.txt and
                        {prefix}_genotype.txt (default: 'output')
  --cn-min CN_MIN       Minimum copy number
  --cn-max CN_MAX       Maximum copy number
  --ado ADO             Allelic dropout rate
  --seq-error SEQ_ERROR Sequencing error rate
  --af AF               Expected allele frequency
  --cn-noise CN_NOISE   Copy-number noise level
  --tree-batch TREE_BATCH
                        Tree batch size
  --node-batch NODE_BATCH
                        Node batch size
  --verbose             Enable verbose output
```
