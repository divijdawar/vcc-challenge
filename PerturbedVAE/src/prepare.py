"""Reproducible observation preparation without fitting response-derived features."""
from __future__ import annotations

import csv
import json
import platform
import re
import shutil
import tempfile
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import sparse

from . import __version__
from .config import config, __file__ as config_file
from .h5ad import CountSource, sha256, unique_ids
from .observations import align, control_pools, log_cp10k


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def gene_csv(path, column):
    with Path(path).open(newline="") as handle:
        return unique_ids([row[column] for row in csv.DictReader(handle)], str(path))


def resolve_data_inputs():
    """Resolve release metadata and output paths using the imported config class."""
    path = Path(config_file).resolve()
    root = (path.parent / config.project_root).resolve()
    output = (root / config.output_dir).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("output_dir must be inside project_root")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("config.seed must be a nonnegative integer")
    if not isinstance(config.control_reference_fraction, (int, float)) or not 0 <= config.control_reference_fraction < 1:
        raise ValueError("control_reference_fraction must be in [0,1)")
    if type(config.chunk_rows) is not int or not 1 <= config.chunk_rows <= 4096:
        raise ValueError("chunk_rows must be an integer in [1, 4096]")
    inputs = {
        "project_root": str(root), "output_dir": str(output), "seed": config.seed,
        "chunk_rows": config.chunk_rows, "gene_identifier_namespace": config.gene_identifier_namespace,
        "normalization_target_sum": config.normalization_target_sum,
        "control_reference_fraction": config.control_reference_fraction,
        "sources": deepcopy(config.sources),
    }
    input_files = [path]
    if config.release_enabled:
        data_dir = (root / config.release_data_dir).resolve()
        manifest_path = data_dir / config.release_manifest
        manifest = json.loads(manifest_path.read_text())
        inputs["release_manifest"] = manifest
        input_files.append(manifest_path)
        if not manifest["contexts"] or len(set(manifest["contexts"])) != len(manifest["contexts"]):
            raise ValueError("Release contexts must be nonempty and unique")
        inputs["output_genes"] = gene_csv(data_dir / config.release_gene_file, config.release_gene_column)
        inputs["challenge_targets"] = gene_csv(data_dir / config.release_target_file, config.release_target_column)
        input_files.extend([data_dir / config.release_gene_file, data_dir / config.release_target_file])
        for context in manifest["contexts"]:
            inputs["sources"].append({
                "id": f"vcc2026_{context}", "kind": "challenge_controls", "dataset": "vcc2026",
                "path": str(data_dir / config.release_file_template.format(context=context)),
                "counts_location": "X", "counts_are_raw": True, "gene_key": "_index",
                "gene_identifier_namespace": config.gene_identifier_namespace,
                "target_key": manifest["pert_col"], "context_key": manifest["context_col"],
                "expected_context": str(context), "control_labels": [manifest["control_label"]],
                "expected_rows": manifest["per_context"][str(context)]["control_cells"],
                "expected_genes": manifest["n_genes"],
            })
    if "output_genes" not in inputs:
        if config.output_genes is None:
            raise ValueError("Set output_genes when no release is enabled")
        inputs["output_genes"] = deepcopy(config.output_genes)
    inputs["output_genes"] = unique_ids(inputs["output_genes"], "output genes")
    # Selection happens after the full-axis normalization; no HVGs are fitted here.
    encoder_genes = config.encoder_genes if config.encoder_genes is not None else inputs["output_genes"]
    inputs["encoder_genes"] = unique_ids(encoder_genes, "encoder genes")
    inputs.setdefault("challenge_targets", [])
    if not inputs["sources"]:
        raise ValueError("Configure at least one source or a release")
    source_ids = []
    for source in inputs["sources"]:
        source_ids.append(source["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source["id"]):
            raise ValueError("Source IDs must contain only letters, digits, underscores, and hyphens")
        if source["gene_identifier_namespace"] != config.gene_identifier_namespace:
            raise ValueError("All adapters must declare the same verified gene identifier namespace")
        source["path"] = str((root / source["path"]).resolve())
        if output == Path(source["path"]) or output in Path(source["path"]).parents:
            raise ValueError("Output directory cannot contain source inputs")
        if not Path(source["path"]).is_file():
            raise ValueError(f"Missing input: {source['path']}")
    unique_ids(source_ids, "source IDs")
    inputs["input_hashes"] = {str(p): sha256(p) for p in input_files}
    return inputs


def prepare_source(source_config, config, directory):
    directory.mkdir()
    source = CountSource(source_config)
    try:
        if "expected_context" in source_config and set(source.contexts) != {source_config["expected_context"]}:
            raise ValueError("Release context labels disagree with manifest")
        if "expected_rows" in source_config and source.shape[0] != source_config["expected_rows"]:
            raise ValueError("Release row count disagrees with manifest")
        if "expected_genes" in source_config and source.shape[1] != source_config["expected_genes"]:
            raise ValueError("Release gene count disagrees with manifest")
        if source.kind == "challenge_controls" and source.genes != config["output_genes"]:
            raise ValueError("Challenge-control gene order disagrees with release gene axis")
        # The measured axis is explicit; zero expression does not make a gene unmeasured.
        _, encoder_mask = align(sparse.csr_matrix((0, len(source.genes))), source.genes, config["encoder_genes"])
        _, output_mask = align(sparse.csr_matrix((0, len(source.genes))), source.genes, config["output_genes"])
        if not encoder_mask.any() or not output_mask.any():
            raise ValueError("Source has no genes overlapping the encoder/output axis; verify identifiers")
        totals = np.zeros(source.shape[0], dtype=np.float64)
        shards = []
        for start in range(0, source.shape[0], config["chunk_rows"]):
            stop = min(start + config["chunk_rows"], source.shape[0])
            raw = source.rows(start, stop)
            # Validate the specified normalization on each block without caching a second full matrix.
            normalized, totals[start:stop] = log_cp10k(raw)
            if not np.isfinite(normalized.data).all():
                raise ValueError("Nonfinite normalized expression")
            if source.kind == "challenge_controls" and np.any(totals[start:stop] > 1_000_000):
                raise ValueError("Challenge controls exceed the count-export library limit")
            name = f"raw_{start:08d}_{stop:08d}.npz"
            sparse.save_npz(directory / name, raw, compressed=True)
            shards.append({"start": start, "stop": stop, "raw": name, "sha256": sha256(directory / name)})
        pools = control_pools(source.cell_ids, source.contexts, source.is_control, totals,
                              kind=source.kind, source_id=source_config["id"], seed=config["seed"],
                              reference_fraction=config["control_reference_fraction"])
        fields = ["row", "cell_id", "source_label", "target", "context", "batch", "is_control", "library_total", "pool"]
        with (directory / "cells.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(source.shape[0]):
                writer.writerow(dict(zip(fields, [i, source.cell_ids[i], source.labels[i], source.targets[i],
                    source.contexts[i], source.batches[i], int(source.is_control[i]), int(totals[i]), pools[i]])))
        manifest = {
            "id": source_config["id"], "kind": source.kind, "dataset": source_config.get("dataset"),
            "input_path": source_config["path"], "input_sha256": sha256(Path(source_config["path"])),
            "input_bytes": Path(source_config["path"]).stat().st_size, "adapter": source_config,
            "storage": source.storage, "shape": list(source.shape),
            "original_source_genes": source.original_genes, "source_genes": source.genes,
            "gene_identifier_namespace": config["gene_identifier_namespace"],
            "encoder_genes": config["encoder_genes"], "output_genes": config["output_genes"],
            "encoder_measurement_mask": encoder_mask.tolist(), "output_measurement_mask": output_mask.tolist(),
            "normalization": {"method": "log1p_scaled_counts", "target_sum": config["normalization_target_sum"],
                              "total_axis": "all_measured_source_genes",
                              "n_total_genes": len(source.genes), "dtype": "float32", "raw_counts_preserved": True},
            "n_control": int(source.is_control.sum()), "n_perturbed": int((~source.is_control).sum()),
            "zero_library_cells": source.cell_ids[totals == 0].tolist(),
            "pool_counts": dict(Counter(pools)), "contexts": sorted(set(source.contexts)),
            "n_encoder_measured": int(encoder_mask.sum()), "n_output_measured": int(output_mask.sum()),
            "unmeasured_output_genes": [g for g, measured in zip(config["output_genes"], output_mask) if not measured],
            "cells_sha256": sha256(directory / "cells.csv"), "shards": shards,
        }
        write_json(directory / "source.json", manifest)
        return {"id": source_config["id"], "manifest_sha256": sha256(directory / "source.json"),
                **{k: manifest[k] for k in ["shape", "n_control", "n_perturbed", "pool_counts", "n_encoder_measured", "n_output_measured"]}}
    finally:
        source.close()


def prepare(config):
    output = Path(config["output_dir"])
    if output.exists():
        raise ValueError(f"Output already exists; choose a new output_dir: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_json(temporary / "resolved_config.json", config)
        summaries = []
        for source in config["sources"]:
            print(f"Preparing {source['id']} (sparse chunks of {config['chunk_rows']} cells)", flush=True)
            summaries.append(prepare_source(source, config, temporary / source["id"]))
        report = {
            "schema_version": 1, "seed": config["seed"], "sources": summaries,
            "versions": {"vcc": __version__, "python": platform.python_version(),
                         **{name: version(name) for name in ["numpy", "scipy", "h5py"]}},
            "code_hashes": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "config_sha256": sha256(temporary / "resolved_config.json"),
            "encoder_axis_policy": "explicit axis; no response-derived feature selection fitted",
            "target_splits_status": "not_frozen; run python -m vcc split before training",
            "supervised_responses_available": any(s["n_perturbed"] for s in summaries),
            "gradient_training_ready": False,
            "unperturbed_observation_source": "vcc2026",
            "perturbed_observation_sources": sorted({s.get("dataset", "unspecified") for s in config["sources"]
                                                       if s["kind"] == "supervised"}),
            "warning": "Freeze target/context splits before fitting HVGs, PCA, effects, or training. No model has been trained.",
        }
        write_json(temporary / "report.json", report)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return report


def validate_prepared(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if sha256(directory / "resolved_config.json") != report["config_sha256"]:
        raise ValueError("Resolved config hash mismatch")
    result = {"valid": True, "sources": [], "scope": "observation cache integrity; not official submission validation"}
    for summary in report["sources"]:
        source_dir = directory / summary["id"]
        if sha256(source_dir / "source.json") != summary["manifest_sha256"]:
            raise ValueError("Source manifest hash mismatch")
        manifest = json.loads((source_dir / "source.json").read_text())
        if sha256(source_dir / "cells.csv") != manifest["cells_sha256"]:
            raise ValueError("Cell metadata hash mismatch")
        if sha256(Path(manifest["input_path"])) != manifest["input_sha256"]:
            raise ValueError("Original input hash mismatch")
        with (source_dir / "cells.csv").open(newline="") as handle:
            cells = list(csv.DictReader(handle))
        if len(cells) != manifest["shape"][0]:
            raise ValueError("Cached row count mismatch")
        unique_ids([c["cell_id"] for c in cells], "cached cell identifiers")
        stop = 0
        for shard in manifest["shards"]:
            path = source_dir / shard["raw"]
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"Raw shard hash mismatch: {path.name}")
            raw = sparse.load_npz(path)
            if shard["start"] != stop or raw.shape != (shard["stop"] - shard["start"], manifest["shape"][1]):
                raise ValueError("Invalid shard ordering/dimensions")
            if (not np.isfinite(raw.data).all() or np.any(raw.data < 0)
                    or not np.equal(raw.data, np.rint(raw.data)).all()):
                raise ValueError("Cached raw counts are invalid")
            normalized, totals = log_cp10k(raw)
            expected = np.asarray([int(c["library_total"]) for c in cells[shard["start"]:shard["stop"]]])
            if not np.array_equal(totals, expected) or not np.isfinite(normalized.data).all():
                raise ValueError("Cached totals/normalization mismatch")
            stop = shard["stop"]
        if stop != len(cells):
            raise ValueError("Incomplete cached rows")
        result["sources"].append({"id": summary["id"], "cells": len(cells), "valid": True})
    return result
