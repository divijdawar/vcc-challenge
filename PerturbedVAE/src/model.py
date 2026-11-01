"""Tinygrad port of the released PerturbedVAE reference core.

Reference: week3ndzZ/PerturbedVAE, commit 12cf29f4410464c15c142c245b171eed2cca4971.
See THIRD_PARTY_NOTICES.md and licenses/PerturbedVAE-MIT.txt.
Masks are an optional VCC extension. This is not the full challenge adaptation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from tinygrad.nn import Linear

from .config import config
from .counts import CountInputs, ExpectedCounts, expected_counts


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    condition_dim: int
    output_dim: int | None = None
    hidden_dim: int = config.hidden_dim
    invariant_dim: int = config.invariant_dim
    responsive_dim: int = config.responsive_dim
    n_layers: int = config.n_layers
    negative_slope: float = config.negative_slope
    append_measurement_mask: bool = config.append_measurement_mask
    sample_logvar_min: float = config.sample_logvar_min
    sample_logvar_max: float = config.sample_logvar_max
    consistent_logvar_clamp: bool = config.consistent_logvar_clamp
    zero_initialize_structure: bool = config.zero_initialize_structure

    def __post_init__(self):
        for name in ("input_dim", "condition_dim", "hidden_dim", "invariant_dim", "responsive_dim", "n_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.responsive_dim < 2 or self.n_layers < 2:
            raise ValueError("responsive_dim and n_layers must be at least 2")
        if self.output_dim is not None and (type(self.output_dim) is not int or self.output_dim < 1):
            raise ValueError("output_dim must be a positive integer")
        if not 0 <= self.negative_slope <= 1 or not self.sample_logvar_min < self.sample_logvar_max:
            raise ValueError("Invalid activation slope or log-variance bounds")

    @property
    def decoded_dim(self):
        return self.output_dim if self.output_dim is not None else self.input_dim


class MLP:
    """n_layers affine layers; leaky ReLU only between affine layers, as upstream."""

    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, slope, device):
        widths = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.fc = [Linear(a, b) for a, b in zip(widths, widths[1:])]
        self.slope = slope
        for layer in self.fc:
            layer.weight = layer.weight.to(device).cast(dtypes.float32).realize()
            layer.bias = layer.bias.to(device).cast(dtypes.float32).realize()

    def __call__(self, value):
        for i, layer in enumerate(self.fc):
            value = layer(value)
            if i < len(self.fc) - 1:
                value = value.leaky_relu(self.slope)
        return value

    def parameters(self):
        return [parameter for layer in self.fc for parameter in (layer.weight, layer.bias)]


@dataclass
class DiagonalGaussian:
    mean: Tensor
    logvar: Tensor

    def sample(self, noise: Tensor, lower=config.sample_logvar_min, upper=config.sample_logvar_max):
        if noise.shape != self.mean.shape or noise.device != self.mean.device:
            raise ValueError("Gaussian noise must match the posterior's shape and device")
        return self.mean + (0.5 * self.logvar.clip(lower, upper)).exp() * noise.cast(dtypes.float32)


@dataclass
class Posterior:
    invariant: DiagonalGaussian
    responsive_noise: DiagonalGaussian


@dataclass
class LatentNoise:
    invariant: Tensor
    responsive: Tensor

    @classmethod
    def fresh(cls, batch_size, config: ModelConfig, device):
        return cls(Tensor.randn(batch_size, config.invariant_dim, device=device, dtype=dtypes.float32),
                   Tensor.randn(batch_size, config.responsive_dim, device=device, dtype=dtypes.float32))


@dataclass
class StructuralLatents:
    invariant: Tensor
    responsive: Tensor
    structural_noise: Tensor
    raw_weights: Tensor
    weights: Tensor
    rhs: Tensor
    residual: Tensor


@dataclass
class ForwardPass:
    reconstruction: Tensor
    posterior: Posterior
    control_invariant: DiagonalGaussian
    latents: StructuralLatents
    control_reconstruction: Tensor | None = None
    control_posterior: Posterior | None = None
    control_latents: StructuralLatents | None = None


@dataclass
class Generation:
    expression: Tensor
    invariant_posterior: DiagonalGaussian
    latents: StructuralLatents


@dataclass
class ResidualPrediction:
    expected: ExpectedCounts
    difference: Tensor
    target_latents: StructuralLatents
    control_latents: StructuralLatents


def lower_triangular_weights(vector: Tensor, dimension: int):
    """Expand lower-triangle entries in row-major order without indexed assignment."""
    if vector.ndim != 2 or vector.shape[1] != dimension * (dimension - 1) // 2:
        raise ValueError("Structural weight vector has the wrong shape")
    rows, offset = [], 0
    for row in range(dimension):
        zeros = Tensor.zeros(vector.shape[0], dimension - row, device=vector.device, dtype=dtypes.float32)
        rows.append(vector[:, offset:offset + row].tanh().cat(zeros, dim=1) if row else zeros)
        offset += row
    return rows[0].stack(*rows[1:], dim=1)


def forward_substitution(weights: Tensor, rhs: Tensor):
    """Differentiably solve (I-W)z=rhs for strictly lower-triangular W in float32."""
    if rhs.ndim != 2 or weights.shape != (rhs.shape[0], rhs.shape[1], rhs.shape[1]):
        raise ValueError("Structural solve dimensions disagree")
    weights, rhs = weights.cast(dtypes.float32), rhs.cast(dtypes.float32)
    solutions = [rhs[:, 0]]
    for row in range(1, rhs.shape[1]):
        previous = solutions[0].stack(*solutions[1:], dim=1)
        solutions.append(rhs[:, row] + (weights[:, row, :row] * previous).sum(axis=1))
    return solutions[0].stack(*solutions[1:], dim=1)


class PerturbedVAE:
    def __init__(self, config: ModelConfig, *, device: str | None = None):
        self.config = config
        self.device = Device.canonicalize(device)
        blocks = config.n_layers - 1
        make = lambda a, b, depth=blocks: MLP(a, b, config.hidden_dim, depth, config.negative_slope, self.device)
        encoder_dim = config.input_dim * (2 if config.append_measurement_mask else 1)
        self.enc_body = make(encoder_dim, config.hidden_dim)
        self.e_v_mu = make(config.hidden_dim, config.responsive_dim)
        self.e_v_logv = make(config.hidden_dim, config.responsive_dim)
        self.e_i_mu = make(config.hidden_dim, config.invariant_dim)
        self.e_i_logv = make(config.hidden_dim, config.invariant_dim)
        self.W_net = make(config.condition_dim, config.responsive_dim * (config.responsive_dim - 1) // 2)
        self.B_from_i_to_v = make(config.invariant_dim, config.responsive_dim, 1)
        self.decoder = make(config.invariant_dim + config.responsive_dim, config.decoded_dim)
        if config.zero_initialize_structure:
            last = self.W_net.fc[-1]
            last.weight.assign(Tensor.zeros_like(last.weight)).realize()
            last.bias.assign(Tensor.zeros_like(last.bias)).realize()

    def parameters(self):
        names = ("enc_body", "e_v_mu", "e_v_logv", "e_i_mu", "e_i_logv", "W_net", "B_from_i_to_v", "decoder")
        return [parameter for name in names for parameter in getattr(self, name).parameters()]

    def description(self):
        return {"implementation": "tinygrad_reference_core", "config": asdict(self.config),
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "device": self.device, "dtype": "float32", "trained": False}

    def _expression(self, x: Tensor, mask: Tensor | None):
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.config.input_dim or x.device != self.device:
            raise ValueError("Expression must have shape [nonempty batch, input_dim] on the model device")
        x = x.cast(dtypes.float32)
        if mask is not None:
            if mask.shape != x.shape or mask.device != self.device:
                raise ValueError("Measurement mask must match expression shape and device")
            x = x * mask.cast(dtypes.float32)
        if self.config.append_measurement_mask:
            if mask is None:
                raise ValueError("This encoder requires an explicit measurement mask")
            x = x.cat(mask.cast(dtypes.float32), dim=1)
        return x

    def _gaussian(self, mean: Tensor, logvar: Tensor):
        if self.config.consistent_logvar_clamp:
            logvar = logvar.clip(self.config.sample_logvar_min, self.config.sample_logvar_max)
        return DiagonalGaussian(mean.cast(dtypes.float32), logvar.cast(dtypes.float32))

    def encode(self, x: Tensor, mask: Tensor | None = None):
        hidden = self.enc_body(self._expression(x, mask))
        return Posterior(self._gaussian(self.e_i_mu(hidden), self.e_i_logv(hidden)),
                         self._gaussian(self.e_v_mu(hidden), self.e_v_logv(hidden)))

    def encode_invariant(self, x: Tensor, mask: Tensor | None = None):
        hidden = self.enc_body(self._expression(x, mask))
        return self._gaussian(self.e_i_mu(hidden), self.e_i_logv(hidden))

    def structural_latents(self, invariant: Tensor, noise: Tensor, condition: Tensor):
        batch = invariant.shape[0]
        if (invariant.shape != (batch, self.config.invariant_dim)
                or noise.shape != (batch, self.config.responsive_dim)
                or condition.shape != (batch, self.config.condition_dim)
                or any(t.device != self.device for t in (invariant, noise, condition))):
            raise ValueError("Latent/condition shapes or devices disagree with the model")
        raw_weights = self.W_net(condition.cast(dtypes.float32))
        weights = lower_triangular_weights(raw_weights, self.config.responsive_dim)
        rhs = self.B_from_i_to_v(invariant) + noise
        responsive = forward_substitution(weights, rhs)
        residual = responsive - (weights * responsive.unsqueeze(1)).sum(axis=2) - rhs
        return StructuralLatents(invariant, responsive, noise, raw_weights, weights, rhs, residual)

    def decode(self, invariant: Tensor, responsive: Tensor):
        # The released decoder concatenates responsive before invariant coordinates.
        return self.decoder(responsive.cat(invariant, dim=1))

    def forward(self, x: Tensor, condition: Tensor, controls: Tensor, *,
                mask: Tensor | None = None, control_mask: Tensor | None = None,
                noise: LatentNoise | None = None, control_condition: Tensor | None = None,
                control_noise: LatentNoise | None = None):
        posterior = self.encode(x, mask)
        control_posterior = self.encode(controls, control_mask) if control_condition is not None else None
        control_invariant = (control_posterior.invariant if control_posterior is not None
                             else self.encode_invariant(controls, control_mask))
        noise = noise if noise is not None else LatentNoise.fresh(x.shape[0], self.config, self.device)
        sample = lambda q, eps: q.sample(eps, self.config.sample_logvar_min, self.config.sample_logvar_max)
        invariant = sample(posterior.invariant, noise.invariant)
        structural_noise = sample(posterior.responsive_noise, noise.responsive)
        latents = self.structural_latents(invariant, structural_noise, condition)
        result = ForwardPass(self.decode(invariant, latents.responsive), posterior, control_invariant, latents)
        if control_posterior is not None:
            control_noise = (control_noise if control_noise is not None
                             else LatentNoise.fresh(controls.shape[0], self.config, self.device))
            control_invariant_sample = sample(control_posterior.invariant, control_noise.invariant)
            control_exogenous = sample(control_posterior.responsive_noise, control_noise.responsive)
            control_latents = self.structural_latents(control_invariant_sample, control_exogenous, control_condition)
            result.control_reconstruction = self.decode(control_invariant_sample, control_latents.responsive)
            result.control_posterior = control_posterior
            result.control_latents = control_latents
        return result

    def generate(self, controls: Tensor, condition: Tensor, *, mask: Tensor | None = None,
                 noise: LatentNoise | None = None):
        """Reference prior generation: no perturbed expression is accepted or accessed."""
        invariant_posterior = self.encode_invariant(controls, mask)
        noise = noise if noise is not None else LatentNoise.fresh(controls.shape[0], self.config, self.device)
        invariant = invariant_posterior.sample(noise.invariant, self.config.sample_logvar_min, self.config.sample_logvar_max)
        # The reference structural-noise prior is N(0,I), independent of condition.
        latents = self.structural_latents(invariant, noise.responsive, condition)
        return Generation(self.decode(invariant, latents.responsive), invariant_posterior, latents)

    def predict(self, controls: Tensor, condition: Tensor, control_condition: Tensor,
                raw_counts: Tensor, supported_mask: Tensor, *, mask: Tensor | None = None,
                library_totals: Tensor | None = None, noise: LatentNoise | None = None,
                count_inputs: CountInputs | None = None, validate=True):
        """Control-only residual prediction, sharing z_i and Gaussian draws in both branches."""
        if raw_counts.shape != (controls.shape[0], self.config.decoded_dim):
            raise ValueError("Source counts must use the checkpoint's output gene axis")
        posterior = self.encode_invariant(controls, mask)
        noise = noise if noise is not None else LatentNoise.fresh(controls.shape[0], self.config, self.device)
        invariant = posterior.sample(noise.invariant, self.config.sample_logvar_min, self.config.sample_logvar_max)
        # Both reference noise priors are N(0,I). A future conditional prior must
        # transform the SAME base noise with its condition-specific mean/variance.
        target = self.structural_latents(invariant, noise.responsive, condition)
        control = self.structural_latents(invariant, noise.responsive, control_condition)
        difference = self.decode(invariant, target.responsive) - self.decode(invariant, control.responsive)
        expectation = expected_counts(raw_counts, difference, supported_mask, library_totals=library_totals,
                                      prepared=count_inputs, validate=validate)
        return ResidualPrediction(expectation, difference, target, control)
