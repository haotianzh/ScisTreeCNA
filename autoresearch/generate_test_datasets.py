"""Generate multiple test datasets for validation."""
import numpy as np
import pandas as pd
import pickle
from scistreecna.simulate import simulate_data
from scistreecna import util

DATASETS = [
    {"n_cells": 50, "n_sites": 50, "seed": 42, "tag": "50c_50s"},
    {"n_cells": 100, "n_sites": 100, "seed": 123, "tag": "100c_100s_seed123"},
    {"n_cells": 100, "n_sites": 200, "seed": 77, "tag": "100c_200s"},
    {"n_cells": 200, "n_sites": 100, "seed": 99, "tag": "200c_100s"},
]

for ds in DATASETS:
    tag = ds["tag"]
    print(f"Generating {tag}...")
    np.random.seed(ds["seed"])
    reads, true_tree, true_genotype = simulate_data(
        n_cells=ds["n_cells"],
        n_sites=ds["n_sites"],
        mode="clt",
        cn_noise=0.05,
        random_seed=ds["seed"],
    )
    print(f"  reads: {reads.shape}, tree nodes: {len(true_tree.get_all_nodes())}")

    # Save CSV
    n_sites, n_cells, _ = reads.shape
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    site_names = [f"site_{i}" for i in range(n_sites)]
    str_data = np.empty((n_sites, n_cells), dtype=object)
    for i in range(n_sites):
        for j in range(n_cells):
            ref, alt, cn = int(reads[i, j, 0]), int(reads[i, j, 1]), int(reads[i, j, 2])
            str_data[i, j] = f"{ref}|{alt}|{cn}"
    df = pd.DataFrame(str_data, index=site_names, columns=cell_names)
    df.to_csv(f"autoresearch/test_{tag}_reads.csv")

    with open(f"autoresearch/test_{tag}_tree.pkl", "wb") as f:
        pickle.dump(true_tree, f)
    np.savetxt(f"autoresearch/test_{tag}_tg.txt", true_genotype, fmt="%d")

print("Done!")
