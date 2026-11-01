"""Captured float32 training with one pre-update status/metrics transfer."""
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass, fields
from itertools import islice
import json
import math
from pathlib import Path

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam

from .batching import ModelBatch
from .config import config
from .counts import CountInputs
from .losses import PerturbedVAELoss
from .model import LatentNoise, PerturbedVAE
from .prefetch import PrefetchIterator


@dataclass(frozen=True)
class RegularizationSchedule:
    planned_steps: int
    warmup_fraction: float = config.warmup_fraction
    curve: str = config.warmup_curve

    def __post_init__(self):
        if self.planned_steps < 1 or not 0 <= self.warmup_fraction <= 1 or self.curve not in ("linear", "cosine"):
            raise ValueError("Invalid regularization schedule")

    def scale(self, completed_steps):
        if completed_steps < 0:
            raise ValueError("Completed steps cannot be negative")
        warmup = self.planned_steps * self.warmup_fraction
        progress = min(completed_steps / warmup, 1.0) if warmup else 1.0
        return progress if self.curve == "linear" else 0.5 * (1.0 - math.cos(math.pi * progress))


@dataclass
class _BatchLayout:
    attributes: tuple
    noises: tuple
    reference: object
    reference_index: int | None
    counts: tuple

    @property
    def key(self):
        reference_key = None if self.reference is None else (self.reference.context, self.reference.scale_squared)
        return self.attributes, self.noises, reference_key, self.reference_index, self.counts

    def unpack(self, inputs):
        kwargs = {name: inputs[value] if tensor else value for name, tensor, value in self.attributes}
        for name, invariant, responsive in self.noises:
            kwargs[name] = LatentNoise(inputs[invariant], inputs[responsive])
        if self.reference is not None:
            # The original reference was validated at construction; only its
            # tensor profile varies during replay. Do not rerun __post_init__.
            reference = copy(self.reference)
            reference.control_profile = inputs[self.reference_index]
            kwargs["population_reference"] = reference
        if self.counts:
            kwargs["count_inputs"] = CountInputs(**{name: inputs[index] for name, index in self.counts})
        return ModelBatch(**kwargs, _population_validated=True)


def _pack_batches(groups, model, *, residual, population):
    inputs, indices, layouts = [], {}, []

    def add(tensor):
        if id(tensor) not in indices:
            indices[id(tensor)] = len(inputs)
            inputs.append(tensor)
        return indices[id(tensor)]

    excluded = {"noise", "control_noise", "prediction_noise", "population_reference", "count_inputs", "_population_validated"}
    for group in groups:
        attributes = tuple((field.name, isinstance(value := getattr(group, field.name), Tensor),
                            add(value) if isinstance(value, Tensor) else value)
                           for field in fields(ModelBatch) if field.name not in excluded)
        noises = []
        for name, size, needed in (("noise", group.expression.shape[0], True),
                                   ("control_noise", group.controls.shape[0], residual),
                                   ("prediction_noise", group.controls.shape[0], population)):
            if needed:
                noise = getattr(group, name)
                noise = noise if noise is not None else LatentNoise.fresh(size, model.config, model.device)
                noises.append((name, add(noise.invariant), add(noise.responsive)))
        reference = group.population_reference if population else None
        reference_index = add(reference.control_profile) if reference is not None else None
        counts = tuple((f.name, add(getattr(group.count_inputs, f.name))) for f in fields(CountInputs)) if population else ()
        layouts.append(_BatchLayout(attributes, tuple(noises), reference, reference_index, counts))
    # Canonical contiguous buffers give replay a stable layout. Deduplicate object
    # inputs above; clone distinct views of the same buffer only when necessary.
    inputs = [t.detach().contiguous() for t in inputs]
    Tensor.realize(*inputs)
    seen = set()
    for i, tensor in enumerate(inputs):
        if tensor.uop.base in seen:
            inputs[i] = tensor.clone()
        seen.add(tensor.uop.base)
    Tensor.realize(*inputs)
    signature = (tuple(layout.key for layout in layouts), tuple((t.shape, t.dtype, t.device) for t in inputs))
    return inputs, layouts, signature


class Trainer:
    def __init__(self, model: PerturbedVAE, objective: PerturbedVAELoss, schedule: RegularizationSchedule, *,
                 learning_rate=config.learning_rate, gradient_clip=config.gradient_clip,
                 allow_unpaired_reference_alignment=config.allow_unpaired_reference_alignment,
                 diagnostics_directory: str | Path | None = config.diagnostics_directory,
                 jit=config.training_jit, jit_cache_size=config.jit_cache_size):
        if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("Learning rate and gradient clip must be positive and finite")
        if type(jit) is not bool or type(jit_cache_size) is not int or jit_cache_size < 1:
            raise ValueError("jit must be boolean and jit_cache_size must be positive")
        self.model, self.objective, self.schedule = model, objective, schedule
        self.parameters = model.parameters()
        self.optimizer = Adam(self.parameters, lr=learning_rate)
        self.gradient_clip = gradient_clip
        self.completed_steps = 0
        self.allow_unpaired_reference_alignment = allow_unpaired_reference_alignment
        self.diagnostics_directory = Path(diagnostics_directory) if diagnostics_directory is not None else None
        self.jit, self.jit_cache_size = jit, jit_cache_size
        self._objective_settings = (objective.mode, objective.alignment, objective.mmd_bandwidth,
                                    tuple(vars(objective.weights).items()))
        self._kernels = OrderedDict()
        self._update = TinyJit(self._apply_update) if jit else self._apply_update
        self._metric_names = ("loss", "reconstruction", "responsive_kl", "invariant_kl", "alignment", "sparsity")
        if objective.mode == "residual":
            self._metric_names += ("population",)
        self._health_names = ("max_abs_weight", "max_abs_latent", "max_abs_solve_residual", "invariant_mean_variance")

    def _stop(self, message, diagnostics):
        diagnostics = {"error": message, "completed_steps": self.completed_steps, **diagnostics}
        if self.diagnostics_directory is not None:
            self.diagnostics_directory.mkdir(parents=True, exist_ok=True)
            path = self.diagnostics_directory / f"nonfinite_step_{self.completed_steps:08d}.json"
            path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        raise FloatingPointError(f"{message}; optimizer update stopped; diagnostics={diagnostics}")

    def _gradient_kernel(self, layouts):
        def compute(scale, *inputs):
            self.optimizer.zero_grad()
            rows, statuses = [], []
            for layout in layouts:
                group = layout.unpack(inputs)
                residual = self.objective.mode == "residual"
                forward = group.forward(self.model, reconstruct_controls=residual)
                prediction = group.predict(self.model, validated=True) if residual and self.objective.weights.population > 0 else None
                total, terms = self.objective(forward, group.observed_output,
                                              output_mask=group.output_mask, regularization_scale=scale[0],
                                              control_observed=group.observed_controls,
                                              control_output_mask=group.control_output_mask,
                                              prediction=prediction, response_counts=group.raw_responses,
                                              population_reference=group.population_reference,
                                              population_mask=group.population_mask, validated_batch=True,
                                              prediction_controls=group.raw_controls if group.cross_context_transfer else None)
                metrics = [total.detach()] + [terms[name].detach() for name in self._metric_names[1:]]
                latents = [forward.latents]
                if forward.control_latents is not None:
                    latents.append(forward.control_latents)
                if prediction is not None:
                    latents.extend((prediction.target_latents, prediction.control_latents))
                    statuses.append(prediction.expected.finite)
                health = []
                for name in ("weights", "responsive", "residual"):
                    values = [getattr(latent, name).detach().abs().max() for latent in latents]
                    statuses.extend(v.isfinite() for v in values)
                    health.append(values[0].stack(*values[1:]).max())
                health.append(forward.posterior.invariant.mean.detach().var(axis=0, correction=0).mean())
                rows.append(Tensor.stack(*(metrics + health)))
                (total / len(layouts)).backward()
            gradients = [(p.grad if p.grad is not None else Tensor.zeros_like(p)).contiguous() for p in self.parameters]
            norm = Tensor.stack(*(g.square().sum() for g in gradients)).sum().sqrt().reshape(1).contiguous()
            rows = Tensor.stack(*rows)
            finite = rows.isfinite().all() & norm.isfinite().all() & Tensor.stack(*statuses).all()
            report = finite.cast(dtypes.float32).reshape(1).cat(norm, rows.flatten())
            # One realization lets tinygrad jointly schedule metrics and gradients.
            Tensor.realize(report, norm, *gradients)
            return (report, norm, *gradients)
        return compute

    def _apply_update(self, norm, *gradients):
        factor = (self.gradient_clip / norm.maximum(1e-12)).minimum(1.0)[0]
        for parameter, gradient in zip(self.parameters, gradients):
            parameter.grad = gradient * factor
        # Adam schedules all clipped gradients/updates together; no per-parameter realize.
        self.optimizer.step()

    def step(self, groups: list[ModelBatch]):
        """Equal group weighting, fresh explicit noise, clipping, and one Adam update.

        Fixed-shape gradients and optimizer updates are captured separately. One
        packed read between them preserves fail-before-update numerical safety.
        Use jit=False for eager comparison. Input batches are immutable; create a
        new ModelBatch when changing data or prepared masks/counts.
        """
        if not groups:
            raise ValueError("An optimizer step requires at least one target/context group")
        if self.completed_steps >= self.schedule.planned_steps:
            raise ValueError("Planned optimizer-step limit reached")
        if self._objective_settings != (self.objective.mode, self.objective.alignment, self.objective.mmd_bandwidth,
                                        tuple(vars(self.objective.weights).items())):
            raise ValueError("Loss settings are frozen for a Trainer; create a new Trainer after changing them")
        if (self.objective.alignment == "reference_pairwise" and not self.allow_unpaired_reference_alignment
                and any(not group.paired_cells for group in groups)):
            raise ValueError("Unpaired VCC controls require MMD/disabled alignment; explicitly opt in only for reference reproduction")
        population = self.objective.mode == "residual" and self.objective.weights.population > 0
        for group in groups:
            group.prepare_training(population=population)
        scale = self.schedule.scale(self.completed_steps)
        inputs, layouts, signature = _pack_batches(groups, self.model, residual=self.objective.mode == "residual", population=population)
        key = signature
        if key not in self._kernels:
            compute = self._gradient_kernel(layouts)
            self._kernels[key] = TinyJit(compute) if self.jit else compute
            if len(self._kernels) > self.jit_cache_size:
                self._kernels.popitem(last=False)
        self._kernels.move_to_end(key)
        scale_tensor = Tensor([scale], device=self.model.device, dtype=dtypes.float32).realize()
        with Context(TRAINING=1):
            report, norm, *gradients = self._kernels[key](scale_tensor, *inputs)
            # The only device-to-host read in the steady-state training step.
            values = report.numpy()
            names = self._metric_names + self._health_names
            per_group = values[2:].reshape(len(groups), len(names))
            if values[0] != 1.0:
                self._stop("Nonfinite loss, count prediction, structural latent, or accumulated gradients",
                           {"gradient_norm": float(values[1]),
                            "groups": [dict(zip(names, map(float, row))) for row in per_group]})
            self._update(norm, *gradients)
        self.completed_steps += 1
        aggregated = dict(zip(names, map(float, per_group.mean(axis=0))))
        return {**aggregated, "gradient_norm": float(values[1]), "regularization_scale": scale,
                "completed_steps": self.completed_steps}

    def iter_observations(self, observation_steps, vocabulary, *, supported_output_mask=None,
                          population_references=None, prefetch_steps=config.prefetch_steps):
        """Train a lazy iterable of CPU observation-group lists, one list per step.

        Sampling, frozen split allowlists, and verified context mappings stay with
        the caller. CPU reads run ahead; tensor conversion and training run here.
        Supply references keyed by response_context for residual population loss.
        Close the returned generator (or use contextlib.closing) on early exit.
        """
        remaining = self.schedule.planned_steps - self.completed_steps
        source = iter(observation_steps)

        def limited_steps():
            try:
                yield from islice(source, remaining)
            finally:
                if hasattr(source, "close"):
                    source.close()

        with PrefetchIterator(limited_steps(), depth=prefetch_steps) as prefetched:
            for observations in prefetched:
                batches = [ModelBatch.from_observation_group(
                    group, vocabulary, self.model, supported_output_mask=supported_output_mask,
                    population_reference=(population_references or {}).get(group["response_context"]))
                    for group in observations]
                yield self.step(batches)
