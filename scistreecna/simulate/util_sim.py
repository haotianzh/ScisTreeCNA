import numpy as np
from ..util import *


def get_true_tree(nwk, offset=-1):
    tree = util.relabel(util.from_newick(nwk), offset=offset)
    return tree


def load_true_genotype_from_scsim(dirname, i):
    tg = np.loadtxt(f"{dirname}/{i}.tg")
    return tg.astype(int)


def load_tree(dirname, i):
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
    true_hap_file = f"{dirname}/trees_dir/true_haplotypes_dir/true_hap.{i:04}"
    haps = []
    cell_names = []
    with open(true_hap_file) as f:
        line = f.readline().strip()
        num_cells, num_sites = int(line.split()[0]), int(line.split()[1])
        num_cells = num_cells // 2 - 1  # exclude outgroup
        for i in range(num_cells):
            cell_name, hap_parental = f.readline().strip().split()
            cell_name, hap_maternal = f.readline().strip().split()
            hap = [f"{hap_parental[j]}{hap_maternal[j]}" for j in range(num_sites)]
            haps.append(hap)
            cell_names.append(cell_name[:-1])
    df = pd.DataFrame(index=cell_names, data=haps)
    return df


def read_scistree_input(filename):
    arr = []
    with open(filename, "r") as f:
        for line in f.readlines()[1:]:
            row = line.strip().split()[1:]
            row = [float(v) for v in row]
            arr.append(row)
    return np.array(arr, dtype=np.float64)


def get_scistreec_input_with_cn(dirname, i):
    vcf = f"{dirname}/results/vcf_dir/vcf.{i:04}"
    data, tg = read_vcf(vcf)
    tree = load_tree(dirname, i)
    return data, tree, tg


def add_copy_number_noise(reads, noise_prob):
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
    noisy_matrix = reads.copy()
    copy_numbers = noisy_matrix[:, :, 2]
    flip_mask = np.random.rand(*copy_numbers.shape) < noise_prob
    noisy_matrix[:, :, 2][flip_mask] = noisy_matrix[:, :, 2][flip_mask] + (
        1 if np.random.rand() > 0.5 else -1
    )
    return noisy_matrix, flip_mask


def add_copy_number_noise3(reads, noise_prob):
    noisy_matrix = reads.copy()
    copy_numbers = noisy_matrix[:, :, 2]
    flip_mask = np.random.rand(*copy_numbers.shape) < noise_prob
    noisy_matrix[:, :, 2][flip_mask] = noisy_matrix[:, :, 2][
        flip_mask
    ] + np.random.choice([-1, 1], size=flip_mask.sum())
    return noisy_matrix, flip_mask


def random_mask_missing(reads, missing_prob):
    missing_matrix = reads.copy()
    copy_numbers = missing_matrix[:, :, 2]
    missing_mask = np.random.rand(*copy_numbers.shape) < missing_prob
    missing_matrix[:, :, 2][missing_mask] = -1
    return missing_matrix
