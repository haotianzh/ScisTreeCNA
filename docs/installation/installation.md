# Installing ScisTreeCNA

:::{warning}
ScisTreeCNA requires an **NVIDIA GPU with CUDA support** and a compatible NVIDIA driver. CPU-only execution is not supported. Conda can install the CUDA runtime libraries, but it cannot replace the GPU or its driver. See the [GPU requirements](gpu.md) before installation.
:::

If you do not have access to a CUDA-capable GPU, you can run the tutorials using the free T4 GPU in [our Colab notebook](https://colab.research.google.com/drive/1roB2pnTBlFvoQtCNn0QDP3NgvGtK97Yl?usp=sharing).

## Recommended: install with conda

Conda can install a compatible CuPy build and CUDA runtime:

```bash
conda create -n scistreecna python=3.12
conda activate scistreecna
conda install -c conda-forge cupy cuda-version=12.8
```

Choose a CUDA version supported by your NVIDIA driver. Then install ScisTreeCNA:

```bash
git clone https://github.com/haotianzh/ScisTreeCNA.git
cd ScisTreeCNA
pip install .
```

## Alternative: use an existing CUDA toolkit

If a CUDA toolkit is already installed, choose **one** extra that matches its major version:

```bash
git clone https://github.com/haotianzh/ScisTreeCNA.git
cd ScisTreeCNA

pip install ".[cuda11x]"  # CUDA 11.x
# OR
pip install ".[cuda12x]"  # CUDA 12.x
# OR
pip install ".[cuda13x]"  # CUDA 13.x
```

## Verify the installation

```bash
python -c "import scistreecna as scna; print(scna.__version__)"
```

A version number confirms that ScisTreeCNA can find CuPy and a CUDA-capable GPU. You can then run the bundled example:

```bash
scistreecna --input ./examples/test_data_reads.csv --output test_output
```