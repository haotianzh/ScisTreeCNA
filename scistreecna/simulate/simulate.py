import os
import subprocess as sp
import numpy as np
import scistree2
import scsim
from .. import util
from . import util_sim


def attach_cell_to_leaf_beta_splitting(tree, alpha=1, beta=1):
    """Distribute cells among a clonal tree's leaves via recursive Beta-splitting.

    At each internal node, draws a split proportion from Beta(alpha, beta) and
    binomially partitions that node's n_cells between its two children (each child
    clamped to >=2). Mutates `node.n_cells` in place; produces unequal clone sizes.
    """
    traversor = scistree2.utils.TraversalGenerator()
    for node in traversor(tree, order="pre"):
        if not node.is_leaf():
            split_proportion = np.random.beta(alpha, beta)
            num = np.random.binomial(node.n_cells, split_proportion)
            num = min(max(2, num), node.n_cells - 2)
            children = node.get_children()
            children[0].n_cells = num
            children[1].n_cells = node.n_cells - num


def attach_cell_to_leaf_uniform_assignment(tree):
    """Distribute cells roughly equally across a clonal tree's leaves.

    Sets each leaf's n_cells to floor(total / n_leaves); the last leaf absorbs the
    remainder. Mutates leaf.n_cells in place.
    """
    n_cells = tree.root.n_cells
    leaves = tree.get_leaves()
    num = n_cells // len(leaves)
    for leaf in leaves[:-1]:
        tree[leaf].n_cells = num
    tree[leaves[-1]].n_cells = n_cells - num * (len(leaves) - 1)


def generalized_genotype_to_binary_genotype(genotype):
    """Collapse a (n_sites, n_cells, 2) [maternal, paternal]-style genotype into a
    binary mutation matrix.

    Uses the second allele channel: any mutant count >1 becomes 1 (present); when
    both allele channels are 0 the entry is marked missing (-1). Returns (n_sites, n_cells).
    """
    bgeno = genotype[:, :, 1]
    mask = (genotype[:, :, 0] == 0) & (genotype[:, :, 1] == 0)
    bgeno[bgeno > 1] = 1
    bgeno[mask] = -1
    return bgeno


def generate_cell_lineage_tree(n_cells=50, scale=1):
    """Generate a random binary cell-lineage tree (one leaf per cell).

    Builds a random binary topology with n_cells leaves and scales all branch
    lengths by `scale` (defaults are otherwise too long). Returns the tree.
    """
    tree = util.get_random_binary_tree(n_cells)
    tree.root.branch = 1
    util.apply_attr_on_tree(
        tree, "branch", lambda x: x.branch * scale
    )  # branch is a way too large at default.
    return tree


def generate_clonal_tree(n_cells, n_clusters, mode="uniform"):
    """Generate a two-level clonal tree: a clone backbone with a per-clone subtree of cells.

    Builds a random clonal tree of `n_clusters` clones, assigns cells to clones
    (mode="uniform" equal split, "beta" Beta-splitting), then replaces each clone
    leaf with its own random binary subtree of cells (short branches, tagged with a
    `cid` clone id). Leaves are named with contiguous integer cell indices.

    Args:
        n_cells: total number of cells (must be >= n_clusters).
        n_clusters: number of clones.
        mode: cell-to-clone assignment scheme ("uniform" or "beta").

    Returns:
        The combined clonal cell-lineage tree.
    """
    assert n_cells >= n_clusters, "wrong cluster assignments."
    # build clonal tree
    clonal_tree = util.get_random_binary_tree(n_clusters)
    clonal_tree.root.branch = 1
    clonal_tree.root.n_cells = n_cells
    if mode == "beta":
        attach_cell_to_leaf_beta_splitting(clonal_tree)
    if mode == "uniform":
        attach_cell_to_leaf_uniform_assignment(clonal_tree)

    # enlarge branches
    util.apply_attr_on_tree(clonal_tree, "branch", lambda x: x.branch)

    # For each clone leaf, build a per-clone subtree of its cells and graft it in
    offset = 0
    cluster_id = 0
    for leaf in clonal_tree.get_leaves():
        leaf = clonal_tree[leaf]
        n_cells = leaf.n_cells
        t = util.get_random_binary_tree(n_cells)
        t.root.branch = 1
        util.apply_attr_on_tree(t, "branch", lambda x: x.branch / 5)  # set clone branch
        util.apply_attr_on_tree(t, "cid", lambda x: cluster_id)  # tag cells with clone id
        # Name this clone's leaves with contiguous global cell indices
        for i, l in enumerate(t.get_leaves()):
            t[l].identifier = str(i + offset)
            t[l].name = str(i + offset)
        # Replace the clone leaf with its subtree, preserving the parent branch
        branch = leaf.branch
        parent = leaf.parent
        parent.remove_child(leaf)
        parent.add_child(t.root)
        t.root.branch = branch
        t.root.parent = parent
        offset += n_cells
        cluster_id += 1
    clonal_tree._update()
    return clonal_tree


def generate_sample_clt(
    tree,
    n_site,
    n_vaiant_per_site=1,
    error=0.01,
    dropout=0.2,
    dropout_cell_variance=0,
    coverage_mean=10,
    coverage_std=5,
    doublet=0,
    recurrent=0,
    rate_cn_gain=0.05,  # 0.05
    rate_cn_loss=0.05,  # 0.01
    beta_binomial=False,
    random_seed=42,
    tmpfile="tmp_tree.nwk",
    executable=None,
):
    """Run the external `scsim` simulator on a cell-lineage tree to generate read data.

    Writes the tree to a Newick file, invokes the scsim binary with the given
    simulation parameters (SNV error/dropout, copy-number gain/loss rates, coverage,
    doublet/recurrence, etc.), and parses scsim's stdout into reads and true genotypes.

    Args:
        tree: input cell-lineage tree (leaves are cells).
        n_site, n_vaiant_per_site, error, dropout, ...: scsim simulation parameters.
        random_seed: RNG seed passed to scsim.
        tmpfile: path for the temporary Newick file.
        executable: scsim binary path (resolved via scsim.get_executable_path if None).

    Returns:
        (data, tg): data is reads array (n_sites, n_cells, 3) = [ref, alt, cn];
        tg is the true generalized genotype array (n_sites, n_cells, 2).
    """
    with scsim.get_executable_path(executable) as exec:
        executable = str(exec)
    assert os.path.exists(executable), "scsim not found."
    tree = util.relabel(tree, offset=1)  # scsim uses 1-based cell labels
    cn = []
    reads_wild = []
    reads_mut = []
    tg = []
    with open(tmpfile, "w") as out:
        out.write(tree.output(branch_length_func=lambda x: x.branch))
    # Invoke scsim; positional args are its simulation parameters
    res = sp.run(
        f"{executable} {tmpfile} {n_site} {n_vaiant_per_site} {error} {dropout} {doublet} {rate_cn_gain} {rate_cn_loss} {recurrent} 0 {dropout_cell_variance} {coverage_mean} {coverage_std} {random_seed} {int(beta_binomial)}",
        shell=True,
        stdout=sp.PIPE,
        text=True,
    )
    # Parse scsim stdout line-by-line for read counts, copy numbers, and genotypes
    for line in res.stdout.splitlines():
        if "Number of sites" in line:
            nsite = int(line.strip().split(":")[1])
        if "Read Count (0,1)" in line:
            # "Read Count (0,1) = (<ref>,<alt>)": strip parens, split on comma
            reads_wild.append(
                int(line.strip().split("=")[1].strip()[1:-1].split(",")[0])
            )
            reads_mut.append(
                int(line.strip().split("=")[1].strip()[1:-1].split(",")[1])
            )
        if "Copy Number =" in line:
            cn.append(int(line.strip().split("=")[1].strip()))
        if "Genotype =" in line:
            # "Genotype = (<g0>,<g1>)": g0 kept as token, g1 as int
            g = line.strip().split("=")[1].strip()[1:-1]
            g0, g1 = (g.split(",")[0]), int(g.split(",")[1])
            tg.append([g0, g1])
    # Reassemble per-SNV/per-cell streams into the (n_snv, n_cells, 3) reads tensor.
    # scsim reports `nsite` copy-number SEGMENTS, each carrying `n_vaiant_per_site`
    # SNVs that SHARE that segment's copy number, so the true SNV count is
    # nsite * n_vaiant_per_site and consecutive blocks of n_vaiant_per_site SNVs
    # share an identical cn column (used to study CN dependency across sites).
    n_snv = nsite * n_vaiant_per_site
    reads_wild = np.array(reads_wild).reshape(n_snv, -1, 1)
    reads_mut = np.array(reads_mut).reshape(n_snv, -1, 1)
    cn = np.array(cn).reshape(n_snv, -1, 1)
    data = np.concatenate([reads_wild, reads_mut, cn], axis=-1)
    tg = np.array(tg, dtype=int).reshape(n_snv, -1, 2)
    # remove copy 0
    # if True:
    #     data[data[:, :, 2] == 0] = 1
    return data, tg


def generate_sample_clone(
    tree,
    n_site,
    n_vaiant_per_site=1,
    error=0.01,
    dropout=0.2,
    dropout_cell_variance=0,
    coverage_mean=10,
    coverage_std=5,
    doublet=0,
    recurrent=0,
    rate_cn_gain=0.1,
    rate_cn_loss=0.1,
    beta_binomial=False,
    random_seed=42,
    tmpfile="tmp_tree.nwk",
    executable=None,
):
    """Run `scsim` on a clonal tree, then enforce clone-level (shared) copy numbers.

    Like generate_sample_clt but for clonal trees: after parsing scsim output, it
    averages the copy-number column within each clone (using leaf `cid` tags) so all
    cells in a clone share one CN profile.

    Args:
        tree: clonal cell-lineage tree (leaves tagged with clone id `cid`).
        n_site, error, dropout, rate_cn_gain, rate_cn_loss, ...: scsim parameters.
        executable: scsim binary path (resolved via scsim.get_executable_path if None).

    Returns:
        (data, tg): reads (n_sites, n_cells, 3) = [ref, alt, cn] with clone-averaged
        CN, and true generalized genotype tg (n_sites, n_cells, 2).
    """
    with scsim.get_executable_path(executable) as exec:
        executable = str(exec)
    assert os.path.exists(executable), "scsim not found."
    tree = util.relabel(tree, offset=1)  # scsim uses 1-based cell labels
    cn = []
    reads_wild = []
    reads_mut = []
    tg = []
    with open(tmpfile, "w") as out:
        out.write(tree.output(branch_length_func=lambda x: x.branch))
    # Invoke scsim; positional args are its simulation parameters
    res = sp.run(
        f"{executable} {tmpfile} {n_site} {n_vaiant_per_site} {error} {dropout} {doublet} {rate_cn_gain} {rate_cn_loss} {recurrent} 0 {dropout_cell_variance} {coverage_mean} {coverage_std} {random_seed} {int(beta_binomial)}",
        shell=True,
        stdout=sp.PIPE,
        text=True,
    )
    # Parse scsim stdout for read counts, copy numbers, and genotypes (see generate_sample_clt)
    for line in res.stdout.splitlines():
        if "Number of sites" in line:
            nsite = int(line.strip().split(":")[1])
        if "Read Count (0,1)" in line:
            reads_wild.append(
                int(line.strip().split("=")[1].strip()[1:-1].split(",")[0])
            )
            reads_mut.append(
                int(line.strip().split("=")[1].strip()[1:-1].split(",")[1])
            )
        if "Copy Number =" in line:
            cn.append(int(line.strip().split("=")[1].strip()))
        if "Genotype =" in line:
            g = line.strip().split("=")[1].strip()[1:-1]
            g0, g1 = (g.split(",")[0]), int(g.split(",")[1])
            tg.append([g0, g1])
    reads_wild = np.array(reads_wild).reshape(nsite, -1, 1)
    reads_mut = np.array(reads_mut).reshape(nsite, -1, 1)
    cn = np.array(cn).reshape(nsite, -1, 1)
    tg = np.array(tg, dtype=int).reshape(nsite, -1, 2)
    # Look up each cell's clone id (cid) from the tree leaves
    n_site, n_cell, _ = cn.shape
    cluster_ids = []
    for i in range(n_cell):
        cluster_ids.append(tree[str(i)].cid)
    cluster_ids = np.array(cluster_ids)
    # Replace each clone's per-cell CN with the clone mean (shared clone CN profile)
    for cid in range(max(cluster_ids) + 1):
        cn[:, cluster_ids == cid] = np.mean(
            cn[:, cluster_ids == cid], axis=1, keepdims=True
        )
    # cn[cn == 0] = 1
    data = np.concatenate([reads_wild, reads_mut, cn], axis=-1)
    return data, tg


def simulate_data(n_cells, n_sites, mode="clt", n_clusters=4, cn_noise=0.05, missing=0, **kwargs):
    """Top-level entry point: generate a synthetic ScisTreeCNA dataset.

    Builds a ground-truth tree (per-cell lineage for mode="clt", clonal for
    mode="clone"), runs scsim to produce reads + true genotype, then optionally adds
    copy-number noise and missing entries.

    Args:
        n_cells, n_sites: dataset dimensions.
        mode: "clt" (cell lineage tree) or "clone" (clonal tree).
        n_clusters: number of clones (mode="clone" only).
        cn_noise: per-entry probability of perturbing a copy number by +/-1.
        missing: per-entry probability of masking a copy number as missing.
        **kwargs: forwarded to the scsim sampling function.

    Returns:
        (reads, tree, tg): reads (n_sites, n_cells, 3) = [ref, alt, cn], the true
        tree, and the true binary genotype matrix.
    """
    assert mode in ["clt", "clone"], f"Only 'clt', and 'clone' are supported yet."

    if mode == "clt":
        tree = generate_cell_lineage_tree(n_cells=n_cells, scale=0.2)
        reads, tg = generate_sample_clt(tree, n_sites, **kwargs)
        tg = util.ggeno_to_bgeno(tg)
    if mode == "clone":
        tree = generate_clonal_tree(n_cells=n_cells, n_clusters=n_clusters)
        reads, tg = generate_sample_clone(tree, n_sites, **kwargs)
        tg = util.ggeno_to_bgeno(tg)
    if cn_noise > 0:
        reads, masks = util_sim.add_copy_number_noise2(
            reads, noise_prob=cn_noise
        )  # add noise to copy numbers
    if missing > 0:
        reads = util.random_mask_missing(reads, missing_prob=missing)

    return reads, tree, tg
