import os
import re
import numpy as np
from .. import util


def write_to_cellphy(matrix):
    nsite, ncell = matrix.shape
    output = f"cellphy_tmp.vcf"
    header = f"""##fileformat=VCFv4.3
##fileDate=NOW
##source=ov2295
##ncell={ncell}
##nsite={nsite}
##reference=NONE
##contig=<ID=1>
##phasing=NO
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phread-scaled genotype likelihoods">
"""
    with open(output, "w") as out:
        out.write(header)
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for i in range(ncell):
            out.write(f"\t{i}")
        out.write("\n")
        for idx, i in enumerate(range(nsite)):
            chrom, coord, ref, alt, af, snp_id = 1, idx, "A", "T", 0.1, idx
            out.write(
                f"{chrom}\t{coord}\t{snp_id}\t{ref}\t{alt}\t.\tPASS\tAF={af}\tGT:PL"
            )
            for j in range(ncell):
                out.write(f"\t{matrix[i, j]}")
            out.write("\n")


# def phred_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, ado=0.2, seqerr=0.01):
#     # Q-phred score = -10 * log10(p)
#     p00, p01, p10, p11 = np.log(1-seqerr), np.log(seqerr), np.log(seqerr), np.log(1-seqerr)
#     l00 = np.exp((ref_counts*p00 + alt_counts*p01).astype(float))
#     l01 = (1-ado)*np.exp((ref_counts*np.log(0.5*np.exp(p00)+0.5*np.exp(p10))).astype(float)+\
#                          (alt_counts*np.log(0.5*np.exp(p01)+0.5*np.exp(p11))).astype(float))+\
#             (0.5*ado)*(np.exp((ref_counts*p00+alt_counts*p10).astype(float))+\
#                        np.exp((ref_counts*p10+alt_counts*p11).astype(float)))
#     l11 = np.exp((ref_counts*p10 + alt_counts*p11).astype(float))
#     q00 = -10 * np.log10(l00)
#     q11 = -10 * np.log10(l11)
#     q01 = -10 * np.log10(l01)
#     q00 = q00.astype(int)
#     q11 = q11.astype(int)
#     q01 = q01.astype(int)
#     return q00, q01, q11


def phred_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, ado=0.2, seqerr=0.01):
    # Q-phred score = -10 * log10(p)
    p00, p01, p10, p11 = (
        np.log(1 - seqerr),
        np.log(seqerr),
        np.log(seqerr),
        np.log(1 - seqerr),
    )
    l00 = np.exp((ref_counts * p00 + alt_counts * p01).astype(np.float128))
    l01 = (1 - ado) * np.exp(
        (ref_counts * np.log(0.5 * np.exp(p00) + 0.5 * np.exp(p10))).astype(np.float128)
        + (alt_counts * np.log(0.5 * np.exp(p01) + 0.5 * np.exp(p11))).astype(
            np.float128
        )
    ) + (0.5 * ado) * (
        np.exp((ref_counts * p00 + alt_counts * p10).astype(np.float128))
        + np.exp((ref_counts * p10 + alt_counts * p11).astype(np.float128))
    )
    l11 = np.exp((ref_counts * p10 + alt_counts * p11).astype(np.float128))
    q00 = -10 * np.log10(l00)
    q11 = -10 * np.log10(l11)
    q01 = -10 * np.log10(l01)
    q00 = q00.astype(int)
    q11 = q11.astype(int)
    q01 = q01.astype(int)
    return q00, q01, q11


def get_ml_gt(ref_counts, alt_counts, ado=0.2):
    # a, b, c = phred_likelihood_with_fn_fp_flat(ref_counts, alt_counts)
    a, b, c = phred_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, ado=ado)
    d = np.concatenate(
        [a[:, :, np.newaxis], b[:, :, np.newaxis], c[:, :, np.newaxis]], axis=-1
    )
    arg_max = np.argmin(d, axis=-1)
    return arg_max


def get_phred_likelihood(a, b, c, ml_gt):
    gts = ["0/0", "0/1", "1/1"]
    n, m = a.shape
    mat = []
    for i in range(n):
        res = []
        for j in range(m):
            if a[i, j] == b[i, j] == c[i, j] == 0:
                res.append(f"./.:{a[i,j]},{b[i,j]},{c[i,j]}")
            else:
                res.append(f"{gts[ml_gt[i,j]]}:{a[i,j]},{b[i,j]},{c[i,j]}")
        mat.append(res)
    return np.array(mat)


# def run_cellphy(dirname, i, executable='/home/haz19024/softwares/cellphy/cellphy.sh'):
#     os.system(f"{executable} FAST -a -t 30 -r {dirname}/{i}.vcf > {dirname}/{i}.cellphy.log 2>&1")
#     with open(f'{dirname}/{i}.vcf.raxml.bestTree') as f:
#         tree = f.readline().strip()
#     tree = util.from_newick(tree)
#     tree = util.relabel(tree, offset=-1)
#     return tree


def infer_cellphy_tree(
    reads,
    executable,
    tempfile="cellphy_tmp",
    cell_names=None,
):
    assert os.path.exists(executable), "CellPhy not found."
    n_sites = reads.shape[0]
    n_cells = reads.shape[1]
    ref_cnts = reads[:, :, 0]
    alt_cnts = reads[:, :, 1]
    a, b, c = phred_likelihood_with_ado_seqerr_gt(ref_cnts, alt_cnts)

    gt = get_ml_gt(ref_cnts, alt_cnts)
    res = get_phred_likelihood(a, b, c, gt)
    write_to_cellphy(res)
    os.system(f"{executable} FAST -t 30 -r {tempfile}.vcf > {tempfile}.log 2>&1")
    # os.system(f'{executable} SEARCH -t 30 -r cellphy_tmp.vcf > cellphy_tmp.log 2>&1')
    with open(f"cellphy_tmp.vcf.raxml.bestTree") as f:
        tree = f.readline().strip()
    tree = util.from_newick(tree)
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    tree = util.relabel(
        tree, name_map={str(i): name for i, name in enumerate(cell_names)}
    )
    geno = get_cellphy_genotype("cellphy_tmp", n_sites=n_sites, n_cells=n_cells)
    return tree, geno


def read_cellphy_mutation_list(file):
    edges = {}
    with open(file, "r") as f:
        for line in f.readlines():
            items = line.strip().split()
            if int(items[1]) > 0:
                eid, num_muts, muts = items
                ms = []
                for _ in muts.split(","):
                    if ":" in _:
                        ms.append(int(_.split(":")[1]))
                    else:
                        ms.append(int(_))
                edges[eid] = ms
            else:
                eid = items[0]
                edges[eid] = []
    return edges


def read_cellphy_mutation_tree(file):
    with open(file, "r") as f:
        line = f.readline()
        line = line.strip()
    return util.from_newick(line)


def get_cellphy_genotype(prefix, n_cells, n_sites):
    raxml_tree = f"{prefix}.vcf.Mapped.raxml.mutationMapTree"
    raxml_list = f"{prefix}.vcf.Mapped.raxml.mutationMapList"
    if not os.path.exists(raxml_tree) or not os.path.exists(raxml_list):
        print("error!")
        return None
    genotype = np.zeros([n_sites, n_cells], dtype=int)
    edges = read_cellphy_mutation_list(raxml_list)
    tree = read_cellphy_mutation_tree(raxml_tree)
    generator = util.TraversalGenerator(order="post")
    for node in generator(tree):
        if not node.is_root():
            eid = re.search(r"\[(\d+)\]", node.branch).group(1)
            muts = edges[eid]
            muts = [mut - 1 for mut in muts]  # to 0-indexed
            leaves = [int(leaf.name) - 1 for leaf in node.get_leaves()]  # to 0-indexed
            genotype[np.ix_(muts, leaves)] = 1 - genotype[np.ix_(muts, leaves)]
            # genotype[np.ix_(muts, leaves)] = 1
    return genotype
