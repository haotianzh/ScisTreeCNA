from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo
import numpy as np
from io import StringIO
from sklearn.cluster import KMeans
from .. import util


def infer_copy_number_tree(copy_numbers, cell_names=None):
    """
    Constructs a Neighbor-Joining (NJ) tree using the copy number data from reads[:, :, 2]
    and returns the tree in Newick format with only leaf labels (no branch lengths or internal node labels).

    Parameters:
        copy_numbers (numpy.ndarray): A 2D array of shape (n_sites, n_cells), where each column
                                       represents the copy number profile of a cell.

    Returns:
        str: The Newick string representation of the NJ tree with only leaf labels.
    """
    # Extract the number of cells
    n_cells = copy_numbers.shape[1]

    # Compute pairwise distances using only the lower triangle
    pairwise_distances = []
    for i in range(n_cells):
        row_distances = []
        for j in range(i):  # Only compute distances for the lower triangle (j < i)
            distance = np.linalg.norm(copy_numbers[:, i] - copy_numbers[:, j])
            row_distances.append(distance)
        pairwise_distances.append(row_distances)

    # Convert the lower triangle distances to Biopython's DistanceMatrix format
    names = [f"{i}" for i in range(n_cells)]
    distance_matrix = DistanceMatrix(names)

    # Fill the lower triangle of the DistanceMatrix
    for i, row_distances in enumerate(pairwise_distances):
        for j, distance in enumerate(row_distances):
            distance_matrix[i, j] = distance

    # Construct the NJ tree
    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(distance_matrix)
    for clade in nj_tree.get_nonterminals():
        clade.name = None
    newick_string = nj_tree.format("newick")
    import re

    newick = re.sub(r":[^,);]+", "", newick_string)
    if cell_names is None:
        cell_names = util.get_default_cell_names(n_cells)
    tree = util.relabel(
        util.from_newick(newick),
        name_map={str(i): name for i, name in enumerate(cell_names)},
    )
    return tree


def cluster_and_average_copy_numbers(copy_numbers, k=8):
    """
    Cluster cells by their copy number vectors using KMeans and average the copy number
    for each cluster to represent the copy number of each cell in that cluster.

    Parameters:
        reads (numpy.ndarray): A 3D array of shape (n_sites, n_cells, 3), where reads[:, :, 2]
                               contains the copy number data.
        k (int): The number of clusters for KMeans.

    Returns:
        numpy.ndarray: A 2D array of shape (n_sites, n_cells) with the averaged copy numbers
                       for each cell based on its cluster.
    """
    # Transpose to get cell-wise copy number vectors (n_cells x n_sites)
    copy_number_vectors = copy_numbers.T  # Shape: (n_cells, n_sites)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(copy_number_vectors)

    # Compute the average copy number for each cluster
    averaged_copy_numbers = np.zeros_like(copy_numbers.T)  # Shape: (n_cells, n_sites)
    for cluster in range(k):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_average = np.mean(copy_number_vectors[cluster_indices], axis=0)
        averaged_copy_numbers[cluster_indices] = cluster_average

    # Transpose back to original shape (n_sites x n_cells)
    return averaged_copy_numbers.T
