import os

# avoid segfault when using linear solver
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# check CuPy installation
try:
    import cupy
except ImportError:
    raise ImportError(
        "The 'scistreecna' package requires CuPy for GPU acceleration. "
        "Please refer to 'https://github.com/haotianzh/ScisTreeCNA?tab=readme-ov-file#-installation' for your specific setup."
    )

# check GPU availability
try:
    gpu_count = cupy.cuda.runtime.getDeviceCount()
    if gpu_count == 0:
        raise EnvironmentError(
            "No **NVIDIA GPU** detected. 'your_package' requires a CUDA-enabled "
            "GPU to run. Please ensure your hardware and drivers are correctly configured."
        )
except cupy.cuda.runtime.CUDARuntimeError as e:
    raise EnvironmentError(
        f"CUDA Runtime Error detected: {e}. Cupy cannot initialize the CUDA environment. "
        "This usually indicates a problem with the **NVIDIA drivers** or **CUDA Toolkit** setup."
    )

from .cn_estimate import CNEstimator
from .scistreecna import (
    NodeBatchLoader,
    TreeBatchLoader,
    ScisTreeCNA,
    construct_genotype,
    estimate_copy_number,
    find_copy_gain_loss_on_branch,
    infer,
    evaluate,
    map_copy_gain_and_loss,
    set_cuda_device,
    console,
)
from .topological_sort import batch_topological_sort, topological_sort
from .transition_solver import TransitionProbability


__version__ = "0.1.0"  # first version
