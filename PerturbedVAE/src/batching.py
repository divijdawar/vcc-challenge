"""Boundary between prepared NumPy observations and the tinygrad model."""
from dataclasses import dataclass, field

import numpy as np

from tinygrad import Tensor, dtypes

from .conditioning import PerturbationVocabulary
from .counts import CountInputs, count_inputs_from_numpy, prepare_count_inputs
from .model import LatentNoise, PerturbedVAE
from .population import PopulationReference


@dataclass
class ModelBatch:
    expression: Tensor
    condition: Tensor
    controls: Tensor
    observed_output: Tensor
    input_mask: Tensor | None = None
    control_mask: Tensor | None = None
    output_mask: Tensor | None = None
    noise: LatentNoise | None = None
    paired_cells: bool = False
    control_condition: Tensor | None = None
    observed_controls: Tensor | None = None
    control_output_mask: Tensor | None = None
    raw_controls: Tensor | None = None
    raw_responses: Tensor | None = None
    control_library_totals: Tensor | None = None
    supported_mask: Tensor | None = None
    population_reference: PopulationReference | None = None
    population_mask: Tensor | None = None
    prediction_condition: Tensor | None = None
    control_noise: LatentNoise | None = None
    prediction_noise: LatentNoise | None = None
    cross_context_transfer: bool = False
    count_inputs: CountInputs | None = None
    _population_validated: bool = field(default=False, repr=False)

    @classmethod
    def from_observation_group(cls, group: dict, vocabulary: PerturbationVocabulary, model: PerturbedVAE, *,
                               supported_output_mask=None, population_reference: PopulationReference | None = None):
        if group.get("unperturbed_dataset") != "vcc2026":
            raise ValueError("The unperturbed model input must use VCC 2026 controls")
        perturbed, controls = group["perturbed"], group["unperturbed"]
        if perturbed.is_control.any() or not controls.is_control.all():
            raise ValueError("Response and control populations have invalid condition labels")
        if (perturbed.library_totals <= 0).any() or (controls.library_totals <= 0).any():
            raise ValueError("Exclude zero-library cells before creating model batches")
        if not perturbed.output_measurement_mask.any(axis=1).all():
            raise ValueError("Every response cell needs at least one measured output gene")
        if vocabulary.dimension != model.config.condition_dim:
            raise ValueError("Vocabulary and model condition dimensions disagree")
        convert = lambda value: Tensor(value, device=model.device, dtype=dtypes.float32)
        result = cls(convert(perturbed.x), vocabulary.encode(perturbed.targets, device=model.device),
                   convert(controls.x), convert(perturbed.output_x), convert(perturbed.measurement_mask),
                   convert(controls.measurement_mask), convert(perturbed.output_measurement_mask),
                   paired_cells=bool(group.get("paired_cells", False)))
        result.cross_context_transfer = bool(group.get("cross_context_transfer", False))
        if len(set(perturbed.targets)) != 1:
            raise ValueError("One model group must contain a single target")
        target = str(perturbed.targets[0])
        result.control_condition = vocabulary.encode([vocabulary.control_label] * len(controls.x), device=model.device)
        result.prediction_condition = vocabulary.encode([target] * len(controls.x), device=model.device)
        result.observed_controls = convert(controls.output_x)
        result.control_output_mask = convert(controls.output_measurement_mask)
        result.control_library_totals = convert(controls.library_totals)
        raw_controls = None
        if controls.raw_output_counts is not None and perturbed.raw_output_counts is not None:
            raw_controls = controls.raw_output_counts.toarray()
            result.raw_responses = convert(perturbed.raw_output_counts.toarray())
        if supported_output_mask is not None:
            if len(supported_output_mask) != model.config.decoded_dim:
                raise ValueError("Supported-output mask does not match decoder genes")
            support = np.asarray(supported_output_mask, dtype=np.float32)
            if not np.isin(support, (0, 1)).all():
                raise ValueError("Supported-output mask must be binary")
            result.supported_mask = convert(support)
            # Gene identity comes from the frozen output axis, supplied by the caller.
            output_genes = group.get("output_genes")
            if output_genes is None:
                raise ValueError("Group requires output_genes to exclude its own target from population loss")
            if len(output_genes) != model.config.decoded_dim:
                raise ValueError("Group output gene axis does not match the decoder")
            target_mask = np.asarray([0.0 if gene == target else 1.0 for gene in output_genes], dtype=np.float32)
            population_mask = perturbed.output_measurement_mask[0] * support * target_mask
            result.population_mask = convert(population_mask)
            if raw_controls is not None:
                result.count_inputs = count_inputs_from_numpy(raw_controls, support, controls.library_totals, convert)
                result.raw_controls = result.count_inputs.raw
                result.supported_mask = result.count_inputs.mask
                result.control_library_totals = result.count_inputs.totals
            result._population_validated = bool(np.isfinite(population_mask).all() and population_mask.sum() > 0)
        elif raw_controls is not None:
            result.raw_controls = convert(raw_controls)
        if population_reference is not None:
            if population_reference.context != group.get("response_context"):
                raise ValueError("Population reference must match the response source context")
            result.population_reference = population_reference
        return result

    def forward(self, model: PerturbedVAE, *, reconstruct_controls=False):
        return model.forward(self.expression, self.condition, self.controls, mask=self.input_mask,
                             control_mask=self.control_mask, noise=self.noise,
                             control_condition=self.control_condition if reconstruct_controls else None,
                             control_noise=self.control_noise)

    def prepare_training(self, *, population=False):
        """One-time validation/precomputation. Treat batch input tensors as immutable."""
        if not population:
            return
        if any(value is None for value in (self.raw_controls, self.raw_responses, self.supported_mask,
                                          self.population_reference, self.population_mask,
                                          self.control_condition, self.prediction_condition)):
            raise ValueError("Population training needs counts, conditions, reference, and frozen masks")
        if self.count_inputs is None:
            self.count_inputs = prepare_count_inputs(self.raw_controls, self.supported_mask,
                                                     library_totals=self.control_library_totals)
            self.control_library_totals = self.count_inputs.totals
        if self.count_inputs.raw is not self.raw_controls or self.count_inputs.mask is not self.supported_mask:
            raise ValueError("Create a new batch when replacing its prepared count inputs")
        if not self._population_validated:
            valid = (self.population_mask.isfinite().all() & (self.population_mask.sum() > 0)
                     & ((self.population_mask == 0) | (self.population_mask == 1)).all())
            if not bool(valid.item()):
                raise ValueError("Population loss needs a nonempty binary measured, supported, non-target mask")
            self._population_validated = True

    def predict(self, model: PerturbedVAE, *, validated=False):
        if any(value is None for value in (self.control_condition, self.raw_controls,
                                          self.supported_mask, self.prediction_condition)):
            raise ValueError("Residual count training needs control counts/conditions and a frozen support mask")
        return model.predict(self.controls, self.prediction_condition, self.control_condition,
                             self.raw_controls, self.supported_mask, mask=self.control_mask,
                             library_totals=self.control_library_totals, noise=self.prediction_noise,
                             count_inputs=self.count_inputs if validated else None, validate=not validated)
