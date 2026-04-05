"""Generate simulated data for autoresearch benchmarking."""
import numpy as np
import pandas as pd
import pickle
from scistreecna.simulate import simulate_data
from scistreecna import util

N_CELLS = 100
N_SITES = 100
SEED = 42

np.random.seed(SEED)

print(f"Generating simulated data: {N_CELLS} cells, {N_SITES} sites...")
reads, true_tree, true_genotype = simulate_data(
    n_cells=N_CELLS,
    n_sites=N_SITES,
    mode="clt",
    cn_noise=0.05,
    random_seed=SEED,
)

print(f"Reads shape: {reads.shape}")
print(f"True genotype shape: {true_genotype.shape}")
print(f"True tree nodes: {len(true_tree.get_all_nodes())}")
print(f"True tree leaves: {len(true_tree.get_leaves())}")

# Save reads as CSV in the same "ref|alt|cn" format as test_data_reads.csv
cell_names = [f"cell_{i}" for i in range(N_CELLS)]
site_names = [f"site_{i}" for i in range(N_SITES)]

n_sites, n_cells, _ = reads.shape
str_data = np.empty((n_sites, n_cells), dtype=object)
for i in range(n_sites):
    for j in range(n_cells):
        ref, alt, cn = int(reads[i, j, 0]), int(reads[i, j, 1]), int(reads[i, j, 2])
        str_data[i, j] = f"{ref}|{alt}|{cn}"

df = pd.DataFrame(str_data, index=site_names, columns=cell_names)
df.to_csv("autoresearch/sim_data_reads.csv")

# Save true tree
with open("autoresearch/sim_data_tree.pkl", "wb") as f:
    pickle.dump(true_tree, f)

# Save true genotype
np.savetxt("autoresearch/sim_data_tg.txt", true_genotype, fmt="%d")

print("Saved:")
print("  autoresearch/sim_data_reads.csv")
print("  autoresearch/sim_data_tree.pkl")
print("  autoresearch/sim_data_tg.txt")
print("Done!")
