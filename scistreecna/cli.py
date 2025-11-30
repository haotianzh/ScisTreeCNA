import argparse
import sys
import os
import numpy as np
from .scistreecna import infer, console
from . import util

def main():
    parser = argparse.ArgumentParser(
        description="CLI for ScisTreeCNA inference."
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input reads file (.csv format, see example)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Prefix for output files. Saves as {prefix}_tree.txt and {prefix}_genotype.txt (default: 'output')"
    )   

    # --- Hyperparameters (matching your infer function defaults) ---
    parser.add_argument("--cn-min", type=int, default=1, help="Minimum copy number")
    parser.add_argument("--cn-max", type=int, default=5, help="Maximum copy number")
    parser.add_argument("--ado", type=float, default=0.1, help="Allelic dropout rate")
    parser.add_argument("--seq-error", type=float, default=0.01, help="Sequencing error rate")
    parser.add_argument("--af", type=float, default=0.5, help="Allele Frequency")
    parser.add_argument("--max-iter", type=int, default=0, help="Maximal iteration")
    parser.add_argument("--cn-noise", type=float, default=0.05, help="Copy number noise")
    parser.add_argument("--tree-batch", type=int, default=64, help="Tree batch size")
    parser.add_argument("--node-batch", type=int, default=64, help="Node batch size")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")

    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.input):
        console.print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    try:
        reads, cell_names, site_names = util.read_csv(args.input)
    except Exception as e:
        console.print(f"Error loading .csv: {e}")
        sys.exit(1)

    # 3. Run Inference
    # We pass the parsed arguments into your existing infer function
    try:
        tree, geno = infer(
            reads=reads,
            cell_names=cell_names,
            cn_min=args.cn_min,
            cn_max=args.cn_max,
            ado=args.ado,
            seq_error=args.seq_error,
            af=args.af,
            max_iter=args.max_iter,
            cn_noise=args.cn_noise,
            tree_batch_size=args.tree_batch,
            node_batch_size=args.node_batch,
            verbose=args.verbose
        )
    except Exception as e:
        console.print(f"Error during inference: {e}")
        sys.exit(1)

    # Construct filenames based on prefix
    tree_output_path = f"{args.output}_tree.txt"
    geno_output_path = f"{args.output}_genotype.txt"

    # Save Tree
    try:
        with open(tree_output_path, "w") as f:
            # Assuming the tree object has a __str__ representation (like Newick)
            # If your tree object needs a specific .write() method, change this line.
            f.write(str(tree)) 
        console.print(f"Tree saved to {tree_output_path}")
    except Exception as e:
        console.print(f"Failed to save tree: {e}")

    # Save Genotype
    try:
        # Saving as integer matrix in text format
        np.savetxt(geno_output_path, geno, fmt='%d', delimiter='\t')
        console.print(f"Genotype saved to {geno_output_path}")
    except Exception as e:
        console.print(f"Failed to save genotype: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())