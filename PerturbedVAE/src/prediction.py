"""Reproducible residual prediction and integer export from supplied controls only."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from tinygrad import Tensor, dtypes

from .checkpoint import load_prediction_bundle
from .config import config, __file__ as config_file
from .counts import ExpectedCounts
from .model import LatentNoise
from .observations import ObservationReader, stable_rng
from .prepare import gene_csv
from .submission import SubmissionContract, SubmissionWriter, legal_counts, validate_submission


def sample_integer_counts(raw_counts, expectation: ExpectedCounts, rng):
    """Integer sampling is an export policy, never a differentiable training step."""
    raw = legal_counts(raw_counts, config.max_counts_per_cell).toarray().astype(np.int64)
    probabilities = expectation.probabilities.detach().numpy().astype(np.float64)
    mask = expectation.supported_mask.detach().numpy().astype(bool)
    null_rows = expectation.null_rows.detach().numpy().astype(bool)
    if probabilities.shape != raw.shape or mask.shape != (raw.shape[1],) or null_rows.shape != (raw.shape[0],):
        raise ValueError("Count conversion and raw-count axes disagree")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Nonfinite or negative sampling probabilities")
    output = raw.copy()
    for row in range(len(raw)):
        if null_rows[row]:
            continue  # Exact no-change export, including unsupported genes.
        budget = int(raw[row, mask].sum())
        if budget == 0:
            output[row, mask] = 0
            continue
        p = probabilities[row, mask]
        if not np.isfinite(p.sum()) or p.sum() <= 0:
            raise ValueError("Positive supported budget requires valid probabilities")
        # Renormalize float32 rounding only; invalid probabilities are never repaired.
        p = p / p.sum()
        output[row, mask] = rng.multinomial(budget, p)
    if not np.array_equal(output.sum(axis=1), raw.sum(axis=1)) or not np.array_equal(output[:, ~mask], raw[:, ~mask]):
        raise RuntimeError("Count export failed library-total/unsupported-gene conservation")
    return sparse.csr_matrix(output.astype(np.int32))


def export_predictions(bundle, readers, contract: SubmissionContract, output_path, *, seed=config.seed,
                       batch_rows=config.export_batch_rows):
    if contract.genes != tuple(bundle.output_genes) or contract.control_label != bundle.vocabulary.control_label:
        raise ValueError("Release gene axis/control label does not match the fitted prediction bundle")
    if set(readers) != set(contract.contexts) or batch_rows < 1:
        raise ValueError("Provide exactly the release contexts and a positive export batch size")
    if batch_rows > min(reader.max_batch_rows for reader in readers.values()):
        raise ValueError("Export batch size exceeds a reader's bounded dense-read limit")
    report = {"seed": seed, "prediction_method": "residual_expected_counts_v1", "groups": [],
              "excluded_zero_library_controls": {}, "source_manifests": {},
              "unsupported_output_genes": [g for g, supported in zip(bundle.output_genes, bundle.supported_output_mask) if not supported],
              "checkpoint_weights_sha256": bundle.metadata["weights_sha256"],
              "control_inputs_modified": False}
    eligible_by_context = {}
    for context, reader in readers.items():
        if (reader.manifest["kind"] != "challenge_controls" or reader.manifest.get("dataset") != "vcc2026"
                or reader.genes != bundle.output_genes or reader.output_genes != bundle.output_genes
                or reader.encoder_genes != bundle.encoder_genes):
            raise ValueError("Control reader must match the fitted source/output/encoder gene axes")
        if set(reader.manifest["contexts"]) != {context}:
            raise ValueError("Control source context identity differs from its release label")
        if reader.manifest["normalization"].get("target_sum", 10000.0) != config.normalization_target_sum:
            raise ValueError("Prepared control normalization differs from the prediction count policy")
        eligible = []
        for row, cell in enumerate(reader.cells):
            if cell["is_control"] != "1" or cell["context"] != context:
                raise ValueError("Control source contains a wrong label or context")
            if int(cell["library_total"]) > contract.max_counts_per_cell:
                raise ValueError("Supplied controls are incompatible with the count cap")
            if cell["pool"] == "inference" and int(cell["library_total"]) > 0:
                eligible.append(row)
        if not eligible:
            raise ValueError(f"No nonzero source controls in context {context}")
        eligible_by_context[context] = eligible
        report["excluded_zero_library_controls"][context] = sum(int(cell["library_total"]) == 0 for cell in reader.cells)
        report["source_manifests"][context] = {"id": reader.manifest["id"], "input_sha256": reader.manifest["input_sha256"]}
    writer = SubmissionWriter(output_path, contract)
    model, vocabulary = bundle.model, bundle.vocabulary
    support = Tensor(bundle.supported_output_mask, device=model.device, dtype=dtypes.float32)
    try:
        for context_index, context in enumerate(contract.contexts):
            reader = readers[context]
            for target_index, target in enumerate(contract.targets):
                row_rng = stable_rng(seed, f"{context}:{target}:source_rows")
                count_rng = stable_rng(seed, f"{context}:{target}:integer_counts")
                noise_rng = stable_rng(seed, f"{context}:{target}:latent_noise")
                source_rows = row_rng.choice(eligible_by_context[context], contract.cells_per_target, replace=True)
                fallback = target not in vocabulary.targets
                # Draw the whole small group's noise so results do not depend on export chunk size.
                invariant_noise = noise_rng.standard_normal((contract.cells_per_target, model.config.invariant_dim)).astype(np.float32)
                responsive_noise = noise_rng.standard_normal((contract.cells_per_target, model.config.responsive_dim)).astype(np.float32)
                null_copies = 0
                for start in range(0, contract.cells_per_target, batch_rows):
                    stop = min(start + batch_rows, contract.cells_per_target)
                    batch = reader.rows(source_rows[start:stop])
                    raw = batch.raw_output_counts
                    if fallback:
                        counts = legal_counts(raw, contract.max_counts_per_cell)
                        null_copies += stop - start
                    else:
                        convert = lambda values: Tensor(values, device=model.device, dtype=dtypes.float32)
                        prediction = model.predict(
                            convert(batch.x), vocabulary.encode([target] * (stop - start), device=model.device),
                            vocabulary.encode([vocabulary.control_label] * (stop - start), device=model.device),
                            convert(raw.toarray()), support, mask=convert(batch.measurement_mask),
                            library_totals=convert(batch.library_totals),
                            noise=LatentNoise(convert(invariant_noise[start:stop]), convert(responsive_noise[start:stop])),
                        )
                        counts = sample_integer_counts(raw, prediction.expected, count_rng)
                        null_copies += int(prediction.expected.null_rows.detach().numpy().sum())
                    row_ids = [f"pred_{context_index:02d}_{target_index:04d}_{i:04d}" for i in range(start, stop)]
                    writer.append(counts, context=context, target=target, row_ids=row_ids,
                                  source_cell_ids=batch.cell_ids.tolist())
                report["groups"].append({"context": context, "target": target, "cells": contract.cells_per_target,
                                          "no_change_fallback": fallback, "exact_source_copies": null_copies,
                                          "fallback_reason": "no_training_target_or_usable_target_features" if fallback else None})
        report["validation"] = writer.finish()
    except Exception:
        writer.abort()
        raise
    report_path = Path(output_path).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def release_contract():
    root = (Path(config_file).resolve().parent / config.project_root).resolve()
    data_dir = (root / config.release_data_dir).resolve()
    manifest = json.loads((data_dir / config.release_manifest).read_text())
    genes = gene_csv(data_dir / config.release_gene_file, config.release_gene_column)
    targets = gene_csv(data_dir / config.release_target_file, config.release_target_column)
    cells = manifest.get("cells_per_pert", config.cells_per_target)
    if cells != config.cells_per_target:
        raise ValueError("Configured per-target cell count differs from the release")
    contract = SubmissionContract(tuple(genes), tuple(str(c) for c in manifest["contexts"]), tuple(targets),
                                  cells_per_target=cells, control_label=manifest["control_label"])
    return root, data_dir, contract


def predict_from_config():
    root, data_dir, contract = release_contract()
    if config.prediction_bundle is None:
        raise ValueError("Set config.prediction_bundle to a fitted inference bundle before prediction")
    bundle = load_prediction_bundle(root / config.prediction_bundle, device=config.prediction_device)
    cache = (root / config.output_dir).resolve()
    readers = {}
    for context in contract.contexts:
        directory = (cache / f"vcc2026_{context}").resolve()
        if not directory.is_relative_to(cache):
            raise ValueError("Invalid context cache path")
        if not (directory / "source.json").is_file():
            raise ValueError("Prepared VCC controls are missing; run prepare with the checkpoint's encoder gene axis first")
        readers[context] = ObservationReader(directory)
    destination = (root / config.prediction_output).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Prediction output must remain inside the project")
    report = export_predictions(bundle, readers, contract, destination)
    report["output_path"] = str(destination)
    report["official_validation_command"] = ["vcc", "prep", str(destination), "-g", str(data_dir / config.release_gene_file),
                                              "--perts", str(data_dir / config.release_target_file), "--dry-run"]
    destination.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_from_config():
    root, _, contract = release_contract()
    return validate_submission(root / config.prediction_output, contract)
