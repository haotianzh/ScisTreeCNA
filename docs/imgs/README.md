# Figures

Every file in this folder is currently a **generated placeholder**, not real artwork.
Each one renders a grey box reading `PLACEHOLDER` so it cannot be mistaken for a finished
figure in the built site.

To replace one, just overwrite the PNG with the real image at the same filename. No
Markdown edits are needed — the pages already reference these paths.

| File | Used in | Should show |
| :--- | :--- | :--- |
| `clt.png` | [`index.md`](../index.md) | A small cell lineage tree: leaves are sequenced cells, internal nodes are shared ancestors. The ScisTree2 docs have an equivalent figure that could be reused. |
| `ggeno.png` | [`index.md`](../index.md) | Binary genotype (0/1) contrasted with the generalized genotype $(g_0, g_1)$, showing states like $(2,0)$, $(1,1)$, $(3,1)$, $(0,2)$. |
| `jsc.png` | [`index.md`](../index.md) | The JSC model on a branch: a PM converting a wild-type copy to mutant (at most once per site tree-wide) and CNMs gaining/losing single copies (repeatable). |
| `overview.png` | [`index.md`](../index.md) | The pipeline: read counts + copy numbers → leaf likelihoods → NNI local search → tree + called genotypes. |
| `loh.png` | [`tutorials/allele_specific.ipynb`](../tutorials/allele_specific.ipynb) | Copy-number-neutral LOH: total CN stays at 2 while $(1,1) \to (0,2)$, so the variant allele fraction jumps from ~0.5 to ~1.0. |

Candidate sources for the real versions: figures from `ScisTreeCNA_BIOINFO_.pdf` and its
supplement, and the shared conceptual figures in the
[ScisTree2 docs](https://github.com/yufengwudcs/ScisTree2/tree/python/docs/imgs)
(`clt.png`, `geno.png`, `mutmap.png`).

The script that generated the current placeholders is not checked in; they are meant to
be deleted, not regenerated.
