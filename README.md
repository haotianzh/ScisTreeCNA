# ScisTreeCNA  
**Accurate probabilistic reconstruction of cell lineage trees from SNVs and CNAs**

ScisTreeCNA is a GPU-accelerated Python package for reconstructing cell lineage trees (CLTs) from single-cell DNA sequencing data. It integrates both single-nucleotide variants (SNVs) and copy-number alterations (CNAs) under a unified probabilistic model, providing accurate and scalable inference for single-cell datasets with both SNVs and CNAs information.

ScisTreeCNA relies on **NVIDIA CUDA GPUs** to achieve high performance. CPU-only environments are **not supported**.



## System requirements
- OS: Linux / Windows
- GPU: NVIDIA CUDA GPU with the Compute Capability 3.0 or larger.  
- CUDA Toolkit


## Installation

### 1. Install [CuPy](https://cupy.dev/) with 

ScisTreeCNA uses **CuPy** for GPU computation. The recommended way to install CuPy together with CUDA libraries is via **conda**:

```bash
# Example: install CuPy with CUDA 12.x support
conda install -c conda-forge cupy 
