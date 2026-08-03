# Release notes

:::{note}
Draft. Release dates still need to be filled in, and the entries below were reconstructed
from `scistreecna/__init__.py` and the commit history rather than from a written
changelog. Please correct before publishing.
:::

## Unreleased

Support for **allele-specific copy number** input. Input entries may be written as
`ref|alt|cn_maj|cn_min` (for example from CHISEL) in addition to the original
`ref|alt|cn`. The latent generalized genotype and all the transition/DP machinery are
unchanged; only the leaf emission differs, marginalising over how the wild-type copies
split across the two homologs. No phasing is required. This makes copy-number-neutral LOH
identifiable — see [](tutorials/allele_specific.ipynb).

## Version 1.1.0

Fused CUDA kernels and a contiguous memory layout for the batched likelihood evaluation,
giving a large end-to-end speedup. The changes are internal — inferred trees and
genotypes are unchanged.

- Merged the separate `U` / `U·Φ` / `U·Φ'` allocations into a single contiguous GPU
  buffer, and fused several log-space kernels
- Reworked topology pre-computation, including a fix for a quadratic topological sort
- Derived the down-pass from the reversed up-pass instead of sorting a second time
- Skipped leaf nodes in the down-pass, whose values are never read
- Faster NNI candidate generation via pre-filtering
- Fixed a CUDA grid z-dimension overflow that affected large tree batches
- Added `estimate_batch_sizes` for sizing `tree_batch_size` against available GPU memory

## Version 0.1.0

First release: joint SNV + CNA cell lineage tree reconstruction under the JSC model,
GPU-accelerated batched NNI local search, genotype calling, copy-number event mapping,
and the `scistreecna` command-line interface.
