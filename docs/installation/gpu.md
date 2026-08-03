# GPU requirements and troubleshooting

ScisTreeCNA performs all likelihood calculations on the GPU using CuPy and custom CUDA kernels. **There is no CPU fallback.** This page explains the hardware requirements, setup checks, and common errors.

## Requirements

| Component | Requirement |
| :--- | :--- |
| GPU | NVIDIA GPU with CUDA compute capability $\geq 3.0$ and support from the selected CUDA runtime |
| Driver | NVIDIA driver compatible with the selected CUDA runtime |
| CUDA runtime | Installed with CuPy through conda or provided by an existing CUDA installation |
| Python | $\geq 3.8$ |
| OS | Linux or Windows |

AMD and Intel GPUs, Apple Silicon, macOS, and CPU-only machines are not supported. See the [installation guide](installation.md) for setup instructions.

## Check your setup

### 1. Check whether the driver detects the GPU

```bash
nvidia-smi
```

Your GPU should appear in the output. The **CUDA Version** shown in the upper-right corner is the newest CUDA runtime supported by your driver. It is an upper limit, not the version you must install.

### 2. Check whether CuPy can access the GPU

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.__version__)"
```

The device count should be at least `1`.

### 3. Check whether ScisTreeCNA can access the GPU

```bash
python -c "import scistreecna as scna; print(scna.__version__)"
```

This is the final check. When imported, ScisTreeCNA verifies that CuPy is installed and that at least one CUDA device is available. A version number indicates that the setup is working.

## GPU memory requirements

Memory usage depends mainly on:

```text
n_cells × n_sites × N × tree_batch_size
```

Here, `N` is the number of generalized-genotype states determined by `cn_min` and `cn_max`. You can estimate a suitable batch size using:

```python
import scistreecna as scna

info = scna.estimate_batch_sizes(
    n_cells=200,
    n_sites=500,
    cn_min=1,
    cn_max=5,
)
print(info)
```

This function checks the available GPU memory and recommends a `tree_batch_size`. See the [performance guide](../performance.md) for details.

## Select a GPU

On a multi-GPU machine, ScisTreeCNA uses device `0` by default. To select another device, call `set_cuda_device` before running inference:

```python
import scistreecna as scna

scna.set_cuda_device(1)
```

Alternatively, restrict the GPUs visible to the process.

On Linux:

```bash
CUDA_VISIBLE_DEVICES=1 scistreecna --input reads.csv --output result
```

On Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
scistreecna --input reads.csv --output result
```

ScisTreeCNA does not distribute a single run across multiple GPUs.

## Common errors

:::{dropdown} `ImportError: The 'scistreecna' package requires CuPy for GPU acceleration.`
CuPy is not installed in the active environment. Installing `scistreecna` alone does not install CuPy because the required build depends on the CUDA version. Follow the [installation guide](installation.md).
:::

:::{dropdown} `EnvironmentError: No NVIDIA GPU detected.`
CuPy was imported, but no CUDA device was found. Possible causes include:

- The machine has no NVIDIA GPU.
- `CUDA_VISIBLE_DEVICES` is empty or contains an invalid device index.
- A container was started without GPU access, such as without `--gpus all`.
:::

:::{dropdown} `EnvironmentError: CUDA Runtime Error detected: ...`
CuPy could not initialize CUDA. This usually indicates that the CuPy build is incompatible with the NVIDIA driver. For example, a CUDA 12 CuPy build cannot run with a driver that supports only CUDA 11. Compare the CUDA version reported by `nvidia-smi` with the installed CuPy package, then install a compatible build.
:::

:::{dropdown} `cupy.cuda.memory.OutOfMemoryError`
The dataset or tree batch does not fit in GPU memory. Reduce `tree_batch_size` or use `estimate_batch_sizes` to select a suitable value. See the [performance guide](../performance.md).
:::

## Threading

Importing `scistreecna` sets the following environment variables to `1`:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
```

This prevents a threaded BLAS library from causing failures while SciPy constructs the transition matrices. Because these variables must be set before BLAS is initialized, import `scistreecna` before other numerical packages and do not override these values.