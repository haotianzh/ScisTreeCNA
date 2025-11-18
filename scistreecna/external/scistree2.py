import scistree2 as s2
from .. import util


def infer_scistree2_tree(reads, cell_names=None):
    n_cells = reads.shape[1]
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    gp = s2.probability.from_reads(reads, cell_names=cell_names, posterior=False)
    caller_spr = s2.ScisTree2(threads=8)
    tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
    return tree_spr, imputed_genotype_spr.values
