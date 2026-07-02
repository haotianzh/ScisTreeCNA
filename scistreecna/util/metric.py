import numpy as np


def tree_accuracy(tree1, tree2):
    """Fraction of tree1's splits (clades) that are also present in tree2.
    Splits are represented by their leaf-label sets; only splits containing the
    leaf "1" are compared (a fixed reference leaf used to orient the splits).
    tree1 is the truth, tree2 the inferred tree."""
    count = 0
    splits1 = tree1.get_splits(return_label=True, contains_leaf="1")
    splits2 = tree2.get_splits(return_label=True, contains_leaf="1")
    for split in splits1:
        count += int(split in splits2)
    return count / len(splits1)


def generalized_tree_accuracy(tree1, tree2):
    """Clade-size weighted version of tree_accuracy: each recovered split
    contributes its number of leaves rather than 1, so larger clades count more.
    Returns sum(|split| for recovered splits) / sum(|split| for all true splits)."""
    total = 0
    total_weights = 0
    splits1 = tree1.get_splits(return_label=True)
    splits2 = tree2.get_splits(return_label=True)
    for split in splits1:
        total += int(split in splits2) * len(split)
        total_weights += len(split)
    return total / total_weights


def normalized_rf_distance(tree1, tree2):
    """Normalized Robinson-Foulds distance between two trees.
    RF = |S1 △ S2| / (|S1| + |S2|), where △ is symmetric difference.
    Returns a value in [0, 1]: 0 = identical topology, 1 = completely different."""
    splits1 = tree1.get_splits(return_label=True)
    splits2 = tree2.get_splits(return_label=True)
    if len(splits1) == 0 and len(splits2) == 0:
        return 0.0
    sym_diff = len(splits1 ^ splits2)  # frozenset symmetric difference
    return sym_diff / (len(splits1) + len(splits2))


def split_accuracy(splits1, splits2):
    """Fraction of splits1 that also appear in splits2 (operates on pre-computed
    split sets rather than trees)."""
    count = 0
    for split in splits1:
        count += int(split in splits2)
    return count / len(splits1)


def genotype_accuarcy(geno1, geno2):
    """Fraction of matching genotype calls between two genotype matrices.
    Entries equal to -1 in geno1 (missing/masked) are excluded from both the
    numerator and counted as 0 in the mean. (Name misspelling kept intentionally.)"""
    mask = geno1 != -1
    return np.mean((geno1 == geno2) & mask)


def is_covered(clade1, clade2):
    """Determine the ancestor/descendant relationship between two mutation
    presence vectors (per-cell 0/1 indicators for two SNPs).
    Returns 1 if clade1 (snp1) is ancestral to clade2, -1 if clade2 is ancestral
    to clade1, and 0 if they conflict (neither contains the other)."""
    # diff = clade1 - clade2 per cell: 1 => cell has snp1 not snp2, -1 => snp2 not
    # snp1, 0 => same. Seeing both 1 and -1 (plus a 0) means the clades overlap
    # but neither nests in the other -> no ancestor/descendant relationship.
    diff = clade1 - clade2
    if 0 in diff and 1 in diff and -1 in diff:
        return 0  # no relationship
    if 1 in diff:
        return 1  # clade1(snp1) happens before clade2(snp2)
    else:
        return -1  # clade2(snp2) happens before clade1(snp1)


def get_ancestor_descendant_pairs(geno):
    """Infer ancestor->descendant SNP ordering from a genotype matrix.
    geno has shape (num_snp, num_cells). For every SNP pair their nesting is
    decided by is_covered; returns a dict mapping each SNP "s{i}" to the list of
    SNPs that are its descendants (i.e. mutations occurring later in the lineage)."""
    num_snp, _ = geno.shape
    mutations = {f"s{_}": [] for _ in range(num_snp)}
    for i in range(num_snp):
        for j in range(i + 1, num_snp):
            code = is_covered(geno[i], geno[j])
            if code == 1:
                mutations[f"s{i}"].append(f"s{j}")
            elif code == -1:
                mutations[f"s{j}"].append(f"s{i}")
    return mutations


def ancestor_descendant_error(mutation1, mutation2):
    """Fraction of true ancestor->descendant SNP pairs (from mutation1, the truth)
    that are NOT present as ancestor->descendant in mutation2 (the inferred
    ordering). Both args are dicts as returned by get_ancestor_descendant_pairs."""
    count = 0
    total = 0
    for snp1 in mutation1:
        for snp2 in mutation1[snp1]:
            count += int(snp2 not in mutation2[snp1])
            total += 1
    # total = len(mutation1) * (len(mutation1) - 1) / 2
    return count / total


def different_lineage_error(mutation1, mutation2):
    """Fraction of SNP pairs that lie on different lineages in the truth
    (mutation1: neither is an ancestor of the other) but are wrongly placed in an
    ancestor/descendant relationship in mutation2 (the inferred ordering).
    Each such mis-ordering (in either direction) is counted as an error."""
    count = 0
    total = 0
    muts = list(mutation1.keys())
    num_mut = len(muts)
    for i in range(num_mut):
        for j in range(i + 1, num_mut):
            if muts[i] not in mutation1[muts[j]] and muts[j] not in mutation1[muts[i]]:
                count += int(muts[j] in mutation2[muts[i]]) + int(
                    muts[i] in mutation2[muts[j]]
                )
                total += 1
    return count / total
