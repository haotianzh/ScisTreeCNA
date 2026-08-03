# How to cite ScisTreeCNA

If you use ScisTreeCNA in your work, please cite the following preprint:

> Haotian Zhang and Yufeng Wu. 2025. [**Accurate Probabilistic Reconstruction of Cell Lineage Trees from SNVs and CNAs with ScisTreeCNA**](https://www.biorxiv.org/content/10.1101/2025.11.21.689819v1). *bioRxiv*.

```bibtex
@article{zhang2025scistreecna,
  title   = {Accurate Probabilistic Reconstruction of Cell Lineage Trees
             from {SNVs} and {CNAs} with {ScisTreeCNA}},
  author  = {Zhang, Haotian and Wu, Yufeng},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.11.21.689819},
  url     = {https://www.biorxiv.org/content/10.1101/2025.11.21.689819v1}
}
```

:::{note}
ScisTreeCNA is currently available as a preprint. The citation will be updated when a journal version becomes available.
:::

## Related software

### ScisTree2

**ScisTree2** reconstructs cell lineage trees and calls genotypes from SNVs under the infinite-sites model. ScisTreeCNA uses ScisTree2 to construct the initial tree for local search. If copy-number data are unavailable, ScisTree2 can be used for SNV-based inference.

> Haotian Zhang, Yiming Zhang, Teng Gao, and Yufeng Wu. 2025. [**ScisTree2 enables large-scale inference of cell lineage trees and genotype calling using efficient local search**](https://genome.cshlp.org/content/35/12/2781). *Genome Research* 35: 2781–2791.

- [Source code](https://github.com/yufengwudcs/ScisTree2)
- [Documentation](https://scistree2.readthedocs.io/)

### scsim

**scsim** simulates single-cell read counts under the joint evolution of SNVs and CNAs, including copy-number gains and losses. It is used to generate the benchmark data in the paper.

- [Source code](https://github.com/haotianzh/scsim)



### Experimental analyses

All scripts and resources used to reproduce the analyses in the ScisTreeCNA paper are available in the [ScisTreeCNA analysis repository](https://github.com/haotianzh/scistreecna-analysis-paper).