import numpy as np
import pandas as pd
import pickle
from ..util import *


def get_true_tree(nwk, offset=-1):
    """Parse a Newick string into the internal tree, shifting leaf labels by `offset`
    (default -1, converting 1-based scsim labels to 0-based)."""
    tree = util.relabel(util.from_newick(nwk), offset=offset)
    return tree


def load_true_genotype_from_scsim(dirname, i):
    """Load the integer true-genotype matrix for dataset `i` from `<dirname>/<i>.tg`."""
    tg = np.loadtxt(f"{dirname}/{i}.tg")
    return tg.astype(int)


def load_tree(dirname, i):
    """Load the ground-truth tree for dataset index `i` from a CellCoal-style run dir.

    Reads `<dirname>/results/trees_dir/trees.<i:04>`, drops the outgroup (second
    child of the root), and renames "cell0001"-style leaves to 1-based integer labels.
    """
    tree_file = f"{dirname}/results/trees_dir/trees.{i:04}"
    with open(tree_file) as f:
        nwk = f.readline().strip()
    tree = util.from_newick(nwk)
    tree = BaseTree(root=tree.root.get_children()[1])  # exclude outgroup
    n = len(tree.get_leaves())
    rename_dict = {f"cell{i+1:04}": str(i + 1) for i in range(n)}
    tree = util.relabel(tree, name_map=rename_dict)
    tree.root.set_parent(None)
    return tree


def load_true_genotype_from_cellcoal(dirname, i):
    """Load true haplotypes for dataset `i` from CellCoal's `true_hap.<i:04>` file.

    Reads the diploid haplotype block (two lines per cell: parental + maternal),
    excludes the outgroup, and pairs the two alleles per site into "<p><m>" strings.

    Returns:
        DataFrame indexed by cell name, one column per site, holding diploid genotype
        strings.
    """
    true_hap_file = f"{dirname}/trees_dir/true_haplotypes_dir/true_hap.{i:04}"
    haps = []
    cell_names = []
    with open(true_hap_file) as f:
        line = f.readline().strip()
        num_cells, num_sites = int(line.split()[0]), int(line.split()[1])
        # File lists two haplotypes per cell; halve and drop the outgroup cell
        num_cells = num_cells // 2 - 1  # exclude outgroup
        for i in range(num_cells):
            # Two consecutive lines = parental and maternal haplotypes of one cell
            cell_name, hap_parental = f.readline().strip().split()
            cell_name, hap_maternal = f.readline().strip().split()
            hap = [f"{hap_parental[j]}{hap_maternal[j]}" for j in range(num_sites)]
            haps.append(hap)
            cell_names.append(cell_name[:-1])  # strip trailing parental/maternal suffix
    df = pd.DataFrame(index=cell_names, data=haps)
    return df


def read_scistree_input(filename):
    """Read a ScisTree genotype-probability input file into a float matrix.

    Skips the header line and the leading row label of each line; returns the numeric
    body as a (n_sites, n_cells) float64 array.
    """
    arr = []
    with open(filename, "r") as f:
        for line in f.readlines()[1:]:  # skip header line
            row = line.strip().split()[1:]  # drop leading row label
            row = [float(v) for v in row]
            arr.append(row)
    return np.array(arr, dtype=np.float64)


def get_scistreec_input_with_cn(dirname, i):
    """Load a full ScisTreeCNA input for dataset `i`: reads (with CN) from the VCF and
    the ground-truth tree. Returns (data, tree, tg)."""
    vcf = f"{dirname}/results/vcf_dir/vcf.{i:04}"
    data, tg = read_vcf(vcf)
    tree = load_tree(dirname, i)
    return data, tree, tg


def add_copy_number_noise(reads, noise_prob):
    """Add noise to the copy-number channel (reads[:, :, 2]).

    Each entry flips with prob `noise_prob`: CN==1 -> 2, otherwise CN +/- 1
    (clamped at 0). Returns (noisy_reads, flip_mask). Note: a single random sign is
    drawn per call for the non-one case.
    """
    noisy_matrix = reads.copy()
    copy_numbers = noisy_matrix[:, :, 2]
    flip_mask = np.random.rand(*copy_numbers.shape) < noise_prob
    # noisy_matrix[:, :, 2][flip_mask] = 1 + noisy_matrix[:, :, 2][flip_mask]
    is_one_and_flip = (copy_numbers == 1) & flip_mask
    noisy_matrix[:, :, 2][is_one_and_flip] = 2
    is_two_and_flip = (copy_numbers != 1) & flip_mask
    noisy_matrix[:, :, 2][is_two_and_flip] = noisy_matrix[:, :, 2][is_two_and_flip] + (
        1 if np.random.rand() > 0.5 else -1
    )
    noisy_matrix[noisy_matrix < 0] = 0
    return noisy_matrix, flip_mask


def add_copy_number_noise2(reads, noise_prob):
    """Add copy-number noise: each entry flips with prob `noise_prob` by +/-1 (one
    random sign drawn per call). Returns (noisy_reads, flip_mask). Used by simulate_data."""
    noisy_matrix = reads.copy()
    copy_numbers = noisy_matrix[:, :, 2]
    flip_mask = np.random.rand(*copy_numbers.shape) < noise_prob
    noisy_matrix[:, :, 2][flip_mask] = noisy_matrix[:, :, 2][flip_mask] + (
        1 if np.random.rand() > 0.5 else -1
    )
    return noisy_matrix, flip_mask


def add_copy_number_noise3(reads, noise_prob):
    """Add copy-number noise: each flipped entry shifts by an independently drawn +/-1.
    Returns (noisy_reads, flip_mask)."""
    noisy_matrix = reads.copy()
    copy_numbers = noisy_matrix[:, :, 2]
    flip_mask = np.random.rand(*copy_numbers.shape) < noise_prob
    noisy_matrix[:, :, 2][flip_mask] = noisy_matrix[:, :, 2][
        flip_mask
    ] + np.random.choice([-1, 1], size=flip_mask.sum())
    return noisy_matrix, flip_mask


def random_mask_missing(reads, missing_prob):
    """Mask copy-number entries as missing (-1) with probability `missing_prob`.
    Returns the modified reads array."""
    missing_matrix = reads.copy()
    copy_numbers = missing_matrix[:, :, 2]
    missing_mask = np.random.rand(*copy_numbers.shape) < missing_prob
    missing_matrix[:, :, 2][missing_mask] = -1
    return missing_matrix


def transform_3d_to_2d_dataframe(data_3d: np.ndarray, separator: str = '|') -> pd.DataFrame:
    """Flatten a 3D (m, n, k) array into a 2D (m, n) DataFrame whose cells are the k
    last-axis values joined by `separator` (e.g. "ref|alt|cn")."""
    def concat_elements_to_string(array_1d):
        return separator.join(map(str, array_1d))
    m, n, k = data_3d.shape
    string_list_2d = [
        [concat_elements_to_string(data_3d[i, j, :]) for j in range(n)] 
        for i in range(m)
    ]
    string_array = np.array(string_list_2d, dtype=object)
    df_2d = pd.DataFrame(string_array)
    return df_2d


def save_data(reads, tree, tg, output_prefix):
    """Persist a simulated dataset to disk under `output_prefix`.

    Writes reads as a CSV ("<prefix>_reads.csv", "ref|alt|cn" cells, "s<i>" sites,
    "c<j>" cells), the tree (leaves relabeled to "c<j>") as a pickle
    ("<prefix>_tree.pkl"), and the true genotype as text ("<prefix>_tg.txt").
    """
    n_sites, n_cells, _ = reads.shape
    site_names = [f's{i}' for i in range(n_sites)]
    cell_names = [f'c{i}' for i in range(n_cells)]
    df = transform_3d_to_2d_dataframe(reads)
    df.index = site_names
    df.columns = cell_names
    tree = util.relabel(tree, name_map={str(i): f'c{i}' for i in range(n_cells)})
    df.to_csv(f'{output_prefix}_reads.csv')
    with open(f'{output_prefix}_tree.pkl', 'wb') as out:
        pickle.dump(tree, out)
    np.savetxt(f'{output_prefix}_tg.txt', tg, fmt='%d')

