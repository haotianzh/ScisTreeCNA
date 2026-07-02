import scistree2 as s2
from .. import util


def infer_scistree2_tree(reads, cell_names=None):
    """Run the external ScisTree2 SNV-based method to build the INITIAL tree for
    ScisTreeCNA's local search.

    Uses only the ref/alt read columns (reads[:, :, 0:2]); copy-number is ignored.
    ScisTree2 converts read counts to genotype probabilities and infers a cell
    lineage tree by maximizing genotype likelihood.

    Args:
        reads: array (n_sites, n_cells, 3) = [ref, alt, cn].
        cell_names: leaf labels; defaults to integer cell indices.

    Returns:
        (tree, imputed_genotype): the inferred tree and the imputed binary
        genotype matrix (as a numpy array).
    """
    n_cells = reads.shape[1]
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    gp = s2.probability.from_reads(reads, cell_names=cell_names, posterior=False)
    caller_spr = s2.ScisTree2(threads=8)
    tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
    return tree_spr, imputed_genotype_spr.values
