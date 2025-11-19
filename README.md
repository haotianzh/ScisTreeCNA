# ScisTreeCNA
**Accurate Probabilistic Reconstruction of Cell Lineage Trees from SNVs and CNAs**

ScisTreeCNA is a powerful, **GPU-accelerated** Python package designed for reconstructing **Cell Lineage Trees (CLTs)** from single-cell DNA sequencing data. It addresses a key challenge by integrating both **Single-Nucleotide Variants (SNVs)** and **Copy-Number Alterations (CNAs)** within a unified probabilistic framework. This approach provides accurate and scalable inference, making it ideal for modern single-cell datasets containing both SNV and CNA information.

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
| **Toolkit** | CUDA Runtime | Can be installed later. |
***


## 🚀 Installation

### 1. Install CuPy and CUDA Runtime

ScisTreeCNA uses **CuPy** for its core GPU-accelerated computation. Installing CuPy via `conda-forge` is the easiest and most recommended way to get the necessary **CUDA runtime libraries**:

```bash
# Example: Install CuPy with CUDA 12.x Runtime.
conda install -c conda-forge cupy
```
Note: Installing CuPy with `conda` automatically manages and installs the specific CUDA runtime libraries required for ScisTreeCNA to operate, often simplifying the dependency setup. Check more details at [CuPy official website](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge).



### 2. Install **ScisTreeCNA**

Clone the repository and install the package using `pip`:
```bash
git clone https://github.com/haotianzh/ScisTreeCNA.git
cd ScisTreeCNA
pip install .
```
