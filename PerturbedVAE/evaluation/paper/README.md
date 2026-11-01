# Paper evaluations

Authority: [What Makes a Representation Good for Single-Cell Perturbation
Prediction?, arXiv:2605.19343v1](https://arxiv.org/html/2605.19343v1).
Source checked on 2026-10-30. [specification.json](specification.json) records
metrics, directions, protocols, and unresolved implementation details.

Use Appendix H.2 for the population comparison, Appendix H.1 for synthetic latent
recovery, J.2 for top-20 DE diagnostics, K for Replogle, and L for the MMD ablation.

Reproduction requires resolving the paper's mixed population/cell-level wording.
For each reported result, declare the reduction and any cell pairing. In
particular, arbitrary row pairing between independently sampled populations is
not justified. Appendix H.2 uses regression-based population R-squared; a direct
prediction R-squared must have a different name.

The planned implementation should:

- Require explicit preprocessing and gene-order manifests; do not silently borrow
  the challenge's count normalization for paper reproduction.
- Store per-condition results and the aggregation rule, rather than combining
  single-gene, combinatorial, and synthetic experiments.
- Keep DE readout selection inside scoring, separate from fitted model features.
- Verify metric details against the authors' released code and a fixed fixture
  before labelling a result an exact reproduction.

The reference-code revision in `plan.md` is
`12cf29f4410464c15c142c245b171eed2cca4971`:
[authors' repository](https://github.com/week3ndzZ/PerturbedVAE/tree/12cf29f4410464c15c142c245b171eed2cca4971).
Code-level evaluation parity has not been verified. Details marked unresolved in
the specification are requirements to investigate, not selectable defaults.
