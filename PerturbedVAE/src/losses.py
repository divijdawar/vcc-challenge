"""Reference losses plus an explicitly separate MMD option for unpaired populations."""
from dataclasses import dataclass
import math
from statistics import median

from tinygrad import Tensor, dtypes

from .config import config
from .model import DiagonalGaussian, ForwardPass
from .population import population_effect_loss


@dataclass(frozen=True)
class LossWeights:
    reconstruction: float = config.reconstruction_weight
    responsive_kl: float = config.responsive_kl_weight
    invariant_kl: float = config.invariant_kl_weight
    alignment: float = config.alignment_weight
    sparsity: float = config.sparsity_weight
    population: float = config.population_weight

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("Loss weights must be finite and nonnegative")


def standard_normal_kl(distribution: DiagonalGaussian):
    # Sum over coordinates, average over cells. Responsive KL is in NOISE space.
    return (0.5 * (distribution.mean.square() + distribution.logvar.exp().maximum(1e-12)
                   - 1.0 - distribution.logvar)).sum(axis=1).mean()


def masked_mse(prediction: Tensor, observed: Tensor, mask: Tensor | None = None):
    if prediction.shape != observed.shape or prediction.ndim != 2 or not all(prediction.shape):
        raise ValueError("Reconstruction and observations must have equal, nonempty [cells, genes] shapes")
    squared_error = (prediction - observed.cast(dtypes.float32)).square()
    if mask is None:
        return squared_error.mean()
    if mask.shape != prediction.shape:
        raise ValueError("Output measurement mask must match reconstruction shape")
    mask = mask.cast(dtypes.float32)
    # Adapter rejects all-unmeasured rows. max(1) keeps a degenerate direct call finite.
    return ((squared_error * mask).sum(axis=1) / mask.sum(axis=1).maximum(1.0)).mean()


def reference_alignment(first: Tensor, second: Tensor):
    """Released normalized posterior-mean rowwise MSE; not matched-cell evidence."""
    if first.shape != second.shape:
        raise ValueError("Reference pairwise alignment requires equal population shapes; use MMD for unpaired batches")
    normalize = lambda x: x / x.square().sum(axis=1, keepdim=True).maximum(1e-24).sqrt()
    return (normalize(first) - normalize(second)).square().mean()


def squared_distances(first: Tensor, second: Tensor):
    return (first.unsqueeze(1) - second.unsqueeze(0)).square().sum(axis=2)


def calibrate_mmd_bandwidth(fitting_control_means: Tensor):
    """Call once at initialization on permitted fitting controls, then freeze h."""
    distances = squared_distances(fitting_control_means.detach(), fitting_control_means.detach()).sqrt()
    values = [value for row in distances.tolist() for value in row]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite fitting-control distances")
    positive = [value for value in values if value > 0]
    return max(float(median(positive)), 1e-3) if positive else 1.0


def invariant_mmd(first: Tensor, second: Tensor, bandwidth: float):
    if not math.isfinite(bandwidth) or bandwidth < 1e-3:
        raise ValueError("MMD requires a finite, frozen bandwidth of at least 1e-3")
    def kernel(a, b):
        distances = squared_distances(a, b)
        kernels = [(-distances / (2.0 * h * h)).exp() for h in (bandwidth / 2, bandwidth, bandwidth * 2)]
        return (kernels[0] + kernels[1] + kernels[2]) / 3.0
    return (kernel(first, first).mean() + kernel(second, second).mean()
            - 2.0 * kernel(first, second).mean()).maximum(0.0)


class PerturbedVAELoss:
    def __init__(self, weights: LossWeights | None = None, *, alignment=config.alignment,
                 mmd_bandwidth: float | None = config.mmd_bandwidth, mode=config.training_objective):
        if mode not in ("reference", "residual"):
            raise ValueError("Objective mode must be reference or residual")
        if alignment not in ("reference_pairwise", "mmd", "none"):
            raise ValueError("Unknown alignment mode")
        if alignment == "mmd" and (mmd_bandwidth is None or not math.isfinite(mmd_bandwidth) or mmd_bandwidth < 1e-3):
            raise ValueError("Calibrate and supply a frozen MMD bandwidth before using MMD")
        self.weights = weights or LossWeights()
        self.alignment = alignment
        self.mmd_bandwidth = mmd_bandwidth
        self.mode = mode

    def __call__(self, forward: ForwardPass, observed: Tensor, *, output_mask=None, regularization_scale=1.0,
                 control_observed=None, control_output_mask=None, prediction=None, response_counts=None,
                 population_reference=None, population_mask=None, validated_batch=False, prediction_controls=None):
        if not isinstance(regularization_scale, Tensor) and (not math.isfinite(regularization_scale) or not 0 <= regularization_scale <= 1):
            raise ValueError("Regularization scale must be in [0,1]")
        if isinstance(regularization_scale, Tensor) and regularization_scale.numel() != 1:
            raise ValueError("Regularization scale must be a scalar tensor")
        factual = forward.posterior.invariant.mean
        control = forward.control_invariant.mean
        if self.alignment == "reference_pairwise":
            alignment = reference_alignment(factual, control)
        elif self.alignment == "mmd":
            alignment = invariant_mmd(factual, control, self.mmd_bandwidth)
        else:
            alignment = factual.sum() * 0.0
        terms = {
            "reconstruction": masked_mse(forward.reconstruction, observed, output_mask),
            "responsive_kl": standard_normal_kl(forward.posterior.responsive_noise),
            "invariant_kl": 0.5 * (standard_normal_kl(forward.posterior.invariant)
                                   + standard_normal_kl(forward.control_invariant)),
            "alignment": alignment,
            # Upstream sparsifies the raw W_net vector, before tanh. Preserve that here.
            "sparsity": forward.latents.raw_weights.abs().mean(),
        }
        if self.mode == "residual":
            if forward.control_reconstruction is None or forward.control_posterior is None or control_observed is None:
                raise ValueError("Residual training requires control posterior reconstruction under its control condition")
            control_reconstruction = masked_mse(forward.control_reconstruction, control_observed, control_output_mask)
            terms["reconstruction"] = 0.5 * (terms["reconstruction"] + control_reconstruction)
            terms["responsive_kl"] = 0.5 * (terms["responsive_kl"]
                                            + standard_normal_kl(forward.control_posterior.responsive_noise))
            # Challenge-style sparsity is on effective, bounded lower-triangle entries.
            dimension = forward.latents.weights.shape[1]
            normalization = dimension * (dimension - 1) / 2
            bounded_sparsity = lambda latent: latent.weights.abs().sum(axis=(1, 2)).mean() / normalization
            terms["sparsity"] = 0.5 * (bounded_sparsity(forward.latents) + bounded_sparsity(forward.control_latents))
            if self.weights.population > 0:
                if any(value is None for value in (prediction, response_counts, population_reference, population_mask)):
                    raise ValueError("Population training needs raw responses and a training-only control reference/scale")
                terms["population"] = population_effect_loss(prediction.expected.counts, response_counts,
                                                              population_reference, population_mask,
                                                              validate=not validated_batch,
                                                              prediction_controls=prediction_controls)
            else:
                terms["population"] = terms["reconstruction"] * 0.0
        total = terms["reconstruction"] * self.weights.reconstruction
        for name in ("responsive_kl", "invariant_kl", "alignment", "sparsity"):
            total = total + terms[name] * getattr(self.weights, name) * regularization_scale
        if self.mode == "residual":
            # Population/reconstruction coefficients are fixed throughout regularization warmup.
            total = total + self.weights.population * terms["population"]
        return total, terms
