# PerturbedVAE reference implementation

The tinygrad implementations in `src/model.py` and `src/losses.py` adapt the
architecture, structural map, and loss formulation in:

- Authors' repository: https://github.com/week3ndzZ/PerturbedVAE
- Pinned commit: `12cf29f4410464c15c142c245b171eed2cca4971`
- Reference files: `Norman_double/models/perturbedvae.py` and
  `Norman_double/models/components.py`
- Copyright (c) 2026 Wenkang Jiang
- License: MIT; the original notice is retained in `licenses/PerturbedVAE-MIT.txt`.
- Paper: *What Makes a Representation Good for Single-Cell Perturbation Prediction?*
  https://arxiv.org/html/2605.19343v1

This is a tinygrad port, not the authors' original PyTorch code or a completed
reproduction. Fixed-weight numerical/gradient parity has not been established.

## Reference behavior and explicit differences

The model retains the shared expression backbone; separate Gaussian structural-noise
and invariant heads; strictly lower-triangular, tanh-bounded condition-dependent W;
condition-independent affine invariant-to-responsive map; standard-normal priors;
and responsive-first latent concatenation in the decoder. Each upstream MLP is
built with `n_layers - 1` affine layers, except the one-layer invariant map.

- Differentiable float32 forward substitution replaces the explicit dense solve.
- Externally supplied Gaussian draws are supported, without squeezing batch axes.
- Initialization follows ordinary Linear initialization. Random number streams are
  framework-specific; weights/noise must be supplied identically for future parity.
- Reference sampling clips log variances to [-20,20], while KL uses the unmodified
  log variances and the upstream minimum variance. The optional consistent clamp
  applies the same bounds before sampling and KL; it changes reference behavior.
- Default width 256 and latents (responsive=20, invariant=85) follow the project
  plan. Other constructor settings are explicit in the `src/config.py`.
- Default coefficients (0.01, 0.01, 0.05, 1e-6) and linear warmup follow the project
  plan. They override the released constructor's coefficient defaults and cosine
  warmup. Cosine warmup is available explicitly.
- Reference sparsity is mean absolute raw W logits, as released, rather than
  effective tanh-bounded weights. The residual objective uses the latter.
- Gene masks, concatenated measurement indicators, optional control indicators,
  MMD on invariant means, gradient clipping, and numerical diagnostics are explicit
  project extensions. MMD here is not the separate upstream PerturbedVAEMMD class,
  whose MMD is applied to generated expression.

Residual target/control decoding, shared-noise generation, expected-count
conversion, population-effect training, and sparse integer export are project
extensions. They are separate from the reference reconstruction objective.
ESM features, context descriptors, and learned conditional noise priors remain
future work. This implementation does not inherit identifiability guarantees for
modified losses or cross-dataset control populations.
