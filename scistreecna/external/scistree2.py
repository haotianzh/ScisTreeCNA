import scistree2 as s2


def infer_scistree2_tree(reads):
    n_cells = reads.shape[1]
    gp = s2.probability.from_reads(reads, cell_names=[f'{i}' for i in range(n_cells)], posterior=False)
    caller_spr = s2.ScisTree2(threads=8)
    tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
    return tree_spr, imputed_genotype_spr.values