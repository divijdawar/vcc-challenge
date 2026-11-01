"""One differentiable residual-to-expected-count conversion for training/export."""
from dataclasses import dataclass
import math

from tinygrad import Tensor, dtypes

from .config import config


@dataclass
class CountInputs:
    """Validated, immutable count-side inputs prepared outside captured training."""
    raw: Tensor
    mask: Tensor
    totals: Tensor
    base: Tensor
    budget: Tensor
    fallback: Tensor


def prepare_count_inputs(raw_counts: Tensor, supported_mask: Tensor, *, library_totals=None):
    """Validate once at the boundary; direct prediction/export keeps these checks."""
    if raw_counts.ndim != 2 or not all(raw_counts.shape) or supported_mask.shape != (raw_counts.shape[1],):
        raise ValueError("Supply nonempty [cells, genes] counts and a one-dimensional support mask")
    raw, mask = raw_counts.cast(dtypes.float32), supported_mask.cast(dtypes.float32)
    totals = raw.sum(axis=1) if library_totals is None else library_totals.cast(dtypes.float32)
    if totals.shape != (raw.shape[0],) or any(t.device != raw.device for t in (mask, totals)):
        raise ValueError("Count inputs must share one device and library totals must have shape [cells]")
    valid = (raw.isfinite().all() & (raw >= 0).all() & (raw == raw.round()).all()
             & totals.isfinite().all() & (totals >= raw.sum(axis=1)).all()
             & ((mask == 0) | (mask == 1)).all())
    if not bool(valid.item()):
        raise ValueError("Invalid raw counts, library totals, or support mask")
    scale = config.normalization_target_sum
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_target_sum must be positive and finite")
    base = (1.0 + scale * raw / totals.maximum(1.0).unsqueeze(1)).log()
    budget = (raw * mask).sum(axis=1)
    fallback = raw * mask / budget.maximum(1.0).unsqueeze(1)
    Tensor.realize(base, budget, fallback)
    return CountInputs(raw_counts, supported_mask, totals, base, budget, fallback)


def count_inputs_from_numpy(raw, mask, totals, convert):
    """Prepare count constants before GPU upload, without GPU scalar reads."""
    import numpy as np
    raw, mask, totals = (np.asarray(a, dtype=np.float32) for a in (raw, mask, totals))
    if (raw.ndim != 2 or not all(raw.shape) or mask.shape != (raw.shape[1],)
            or totals.shape != (raw.shape[0],)):
        raise ValueError("Count inputs have incompatible axes")
    if (not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw != np.rint(raw))
            or not np.isfinite(totals).all() or np.any(totals < raw.sum(axis=1))
            or not np.isin(mask, (0, 1)).all()):
        raise ValueError("Invalid raw counts, library totals, or support mask")
    scale = config.normalization_target_sum
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_target_sum must be positive and finite")
    base = np.log(1.0 + scale * raw / np.maximum(totals, 1.0)[:, None])
    budget = (raw * mask).sum(axis=1)
    fallback = raw * mask / np.maximum(budget, 1.0)[:, None]
    return CountInputs(*(convert(a) for a in (raw, mask, totals, base, budget, fallback)))


@dataclass
class ExpectedCounts:
    counts: Tensor
    probabilities: Tensor
    supported_mask: Tensor
    supported_budget: Tensor
    null_rows: Tensor
    log_expression: Tensor
    finite: Tensor


def expected_counts(raw_counts: Tensor, decoder_difference: Tensor, supported_mask: Tensor, *,
                    library_totals: Tensor | None = None, prepared: CountInputs | None = None,
                    validate=True):
    """Keep unsupported counts; redistribute the original supported-count budget.

    raw_counts is on the decoder/output axis. library_totals, when supplied, is
    over the full measured source axis used for normalization. No integer sampling
    or null-copy shortcut enters this differentiable training computation.
    """
    if raw_counts.ndim != 2 or not all(raw_counts.shape) or decoder_difference.shape != raw_counts.shape:
        raise ValueError("Counts and decoder differences need equal nonempty [cells, output genes] shapes")
    if supported_mask.shape != (raw_counts.shape[1],):
        raise ValueError("Supply a frozen one-dimensional supported-output mask")
    if any(t.device != raw_counts.device for t in (decoder_difference, supported_mask)):
        raise ValueError("Count conversion inputs must use one device")
    supplied_prepared = prepared is not None
    if prepared is None:
        if not validate:
            raise ValueError("Captured conversion requires prepared, validated count inputs")
        prepared = prepare_count_inputs(raw_counts, supported_mask, library_totals=library_totals)
    if prepared.raw is not raw_counts or prepared.mask is not supported_mask:
        raise ValueError("Prepared count inputs must belong to the supplied raw counts and support mask")
    if supplied_prepared and library_totals is not None and prepared.totals is not library_totals:
        raise ValueError("Prepared count inputs must use the supplied library totals")
    raw, mask = raw_counts.cast(dtypes.float32), supported_mask.cast(dtypes.float32)
    base, budget, fallback = prepared.base, prepared.budget, prepared.fallback
    scale = config.normalization_target_sum
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_target_sum must be positive and finite")
    expression = (base + decoder_difference).clip(0.0, math.log1p(scale))
    weights = (expression.exp() - 1.0) * mask
    weight_sum = weights.sum(axis=1, keepdim=True)
    probabilities = (weight_sum > 0).where(weights / weight_sum.maximum(1e-12), fallback)
    counts = raw * (1.0 - mask) + probabilities * budget.unsqueeze(1)
    # Check the unclipped difference too: clipping must not hide infinities.
    finite = decoder_difference.isfinite().all() & counts.isfinite().all() & probabilities.isfinite().all()
    if validate and not bool(finite.item()):
        raise FloatingPointError("Nonfinite residual difference, expected counts, or sampling probabilities")
    null_rows = ((decoder_difference * mask) == 0).all(axis=1)
    return ExpectedCounts(counts, probabilities, mask, budget, null_rows, expression, finite)
