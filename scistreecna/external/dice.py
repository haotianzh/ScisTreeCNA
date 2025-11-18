import os 
from .. import simulate
from .. import util


def write_to_dice(reads, output_file="dice_input.tsv"):
    """
    Converts the reads matrix to the DICE input format.

    Parameters:
        reads (numpy.ndarray): A matrix of shape (n_sites, n_cells, 3), where the last dimension contains
                               reference reads, alternate reads, and copy number.
        output_file (str): Path to the output file for DICE input.
    """
    n_sites, n_cells, _ = reads.shape
    chrom = "chr_1"  # Assign a random chromosome name
    start = 0
    end = 1000000  # Arbitrary interval size

    with open(output_file, "w") as f:
        f.write("CELL\tchrom\tstart\tend\tCN states\n")
        for site in range(n_sites):
            for cell in range(n_cells):
                cell_name = f"leaf{cell}"
                cn_state = int(reads[site, cell, 2])  # Extract the copy number state
                f.write(f"{cell_name}\t{chrom}\t{start}\t{end}\t{cn_state}\n")
            start += 1000000
            end += 1000000
    

def infer_dice_tree(reads, executable='/home/haz19024/miniconda3/envs/scistree2/bin/dice', tempfile='dice_tmp'):
    n_cells = reads.shape[1]
    write_to_dice(reads)
    PATH = '/home/haz19024/miniconda3/envs/scistree2/bin/'
    os.system(f'{executable} -i dice_input.tsv -t -o {tempfile} -m balME')
    with open(f'{tempfile}/standard_root_balME_tree.nwk', 'r') as f:
        dice_nwk = f.readline().strip()
    dice_tree = util.from_newick(dice_nwk)
    dice_name_map = {f'leaf{i}': str(i) for i in range(n_cells)}
    dice_tree = util.relabel(dice_tree, name_map=dice_name_map)
    return dice_tree