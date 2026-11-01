"""Training-only control references/scales and differentiable population effects."""
from dataclasses import dataclass, field
import hashlib
import math

import numpy as np
from tinygrad import Tensor

from .config import config
from .observations import ObservationReader, stable_rng


@dataclass
class PopulationReference:
    context: str
    control_profile: Tensor
    scale_squared: float
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.scale_squared) or self.scale_squared < 1e-4:
            raise ValueError("Population scale must be training-fitted and at least 1e-4")
        if self.control_profile.ndim != 1 or not bool(self.control_profile.isfinite().all().item()):
            raise ValueError("Control profile must be a finite gene vector")


def pseudobulk_profile(counts: Tensor):
    bulk = counts.sum(axis=0)
    return (1.0 + config.population_target_sum * bulk / bulk.sum().maximum(1.0)).log()


def population_effect_loss(expected: Tensor, responses: Tensor, reference: PopulationReference, mask: Tensor, *,
                           validate=True, prediction_controls: Tensor | None = None):
    genes = expected.shape[1]
    if (expected.ndim != 2 or responses.ndim != 2 or responses.shape[1] != genes
            or mask.shape != (genes,) or reference.control_profile.shape != (genes,)):
        raise ValueError("Population count, reference, and mask gene axes disagree")
    if validate and not bool((mask.sum() > 0).item()):
        raise ValueError("Population loss needs a measured, supported, non-target output gene")
    if prediction_controls is not None and (prediction_controls.ndim != 2 or prediction_controls.shape[1] != genes):
        raise ValueError("Prediction-control baseline must use the same output gene axis")
    # Cross-context transfer compares effects relative to each context's own
    # baseline. Subtracting the H1 reference from BOTH sides would cancel it and
    # incorrectly train 2026 predictions to reproduce absolute H1 expression.
    predicted_baseline = (pseudobulk_profile(prediction_controls) if prediction_controls is not None
                          else reference.control_profile)
    predicted_effect = pseudobulk_profile(expected) - predicted_baseline
    measured_effect = pseudobulk_profile(responses) - reference.control_profile
    return (((predicted_effect - measured_effect).square() * mask).sum()
            / mask.sum() / reference.scale_squared)


def fit_population_reference(reader: ObservationReader, context: str, permitted_response_rows, *,
                             split_manifest_hash: str, device=None, max_controls=8192):
    """Fit a source-context baseline/scale from explicit frozen-training rows only.

    Never call on development/final response rows. Source-control pools are checked
    here; the target/context exclusions must already be encoded in the allowlist.
    VCC 2026 controls remain the model's unperturbed input, not this source baseline.
    """
    if reader.manifest["kind"] != "supervised" or not split_manifest_hash:
        raise ValueError("A supervised source and frozen split-manifest hash are required")
    allowed = np.asarray(permitted_response_rows)
    if (allowed.ndim != 1 or allowed.dtype.kind not in "iu" or not len(allowed)
            or np.any(allowed < 0) or np.any(allowed >= len(reader.cells))):
        raise ValueError("Supply a nonempty frozen-training row allowlist")
    groups = {}
    for row in np.unique(allowed):
        cell = reader.cells[row]
        if cell["is_control"] != "0" or cell["pool"] != "response_unassigned" or cell["context"] != context:
            raise ValueError("Reference fitting allowlist contains a control, excluded row, or another context")
        groups.setdefault(cell["target"], []).append(int(row))
    controls = [i for i, cell in enumerate(reader.cells)
                if cell["context"] == context and cell["pool"] == "fit" and cell["is_control"] == "1"]
    if not controls or max_controls < 1:
        raise ValueError("Source context needs a nonempty fitting-control pool")
    controls = stable_rng(config.seed, f"population:{reader.manifest['id']}:{context}").permutation(controls)[:max_controls]

    def bulk(rows):
        result = np.zeros(len(reader.output_genes), dtype=np.float64)
        for start in range(0, len(rows), reader.max_batch_rows):
            batch = reader.rows(np.asarray(rows[start:start + reader.max_batch_rows], dtype=np.int64))
            result += np.asarray(batch.raw_output_counts.sum(axis=0, dtype=np.float64)).ravel()
        if result.sum() <= 0:
            raise ValueError("Population has no counts on the output gene axis")
        return np.log1p(config.population_target_sum * result / result.sum())

    control_profile = bulk(controls)
    measured = np.asarray(reader.manifest["output_measurement_mask"], dtype=bool)
    effect_sizes = []
    for target, rows in groups.items():
        mask = measured.copy()
        if target in reader.output_genes:
            mask[reader.output_genes.index(target)] = False
        if not mask.any():
            raise ValueError("Training group has no valid population-effect genes")
        effect = bulk(rows) - control_profile
        effect_sizes.append(float(np.mean(effect[mask] ** 2)))
    fingerprint = lambda rows: hashlib.sha256("\n".join(sorted(reader.cells[int(i)]["cell_id"] for i in rows)).encode()).hexdigest()
    provenance = {"source": reader.manifest["id"], "split_manifest_hash": split_manifest_hash,
                  "fitting_control_ids_sha256": fingerprint(controls), "training_response_ids_sha256": fingerprint(allowed),
                  "n_fitting_controls": len(controls), "n_training_groups": len(groups),
                  "baseline_pool": "fit", "scale_fitted_on": "permitted_training_responses"}
    return PopulationReference(context, Tensor(control_profile.astype(np.float32), device=device),
                               max(float(np.mean(effect_sizes)), 1e-4), provenance)
