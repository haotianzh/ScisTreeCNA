import os 
# avoid segfault when using linear solver
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    console
)
from .topological_sort import batch_topological_sort, topological_sort
from .transition_solver import TransitionProbability



__version__ = "0.1.0"  # first version
