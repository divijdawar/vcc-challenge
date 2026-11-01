"""Sparse normalization, explicit gene alignment, and independent populations."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
from scipy import sparse

from .config import config
from .h5ad import unique_ids


def log_cp10k(raw: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    if not np.isfinite(config.normalization_target_sum) or config.normalization_target_sum <= 0:
        raise ValueError("normalization_target_sum must be positive and finite")
    totals = np.asarray(raw.sum(axis=1, dtype=np.float64)).ravel()
    scale = np.divide(config.normalization_target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    normalized = raw.astype(np.float64).tocsr(copy=True)
    normalized.sum_duplicates()
    normalized.data *= np.repeat(scale, np.diff(normalized.indptr))
    np.log1p(normalized.data, out=normalized.data)
    return normalized.astype(np.float32), totals


class _GeneProjection:
    def __init__(self, source_genes, requested):
        source_genes = unique_ids(source_genes, "source genes")
        requested = unique_ids(requested, "requested genes")
        self.source_size = len(source_genes)
        self.identity = source_genes == requested
        lookup = {g: i for i, g in enumerate(requested)}
        source, destination = [], []
        self.mask = np.zeros(len(requested), dtype=bool)
        for i, gene in enumerate(source_genes):
            if gene in lookup:
                source.append(i)
                destination.append(lookup[gene])
                self.mask[lookup[gene]] = True
        self.projection = sparse.csr_matrix((np.ones(len(source), dtype=np.float32), (source, destination)),
                                            shape=(len(source_genes), len(requested)))

    def __call__(self, matrix):
        if matrix.shape[1] != self.source_size:
            raise ValueError("Gene axis does not match matrix")
        return matrix if self.identity else (matrix @ self.projection).tocsr()


def align(matrix: sparse.csr_matrix, source_genes: list[str], requested: list[str]):
    projection = _GeneProjection(source_genes, requested)
    return projection(matrix), projection.mask.copy()


def stable_rng(seed: int, key: str):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def control_pools(cell_ids, contexts, is_control, totals, *, kind, source_id, seed,
                  reference_fraction=config.control_reference_fraction):
    if not 0 <= reference_fraction < 1:
        raise ValueError("reference_fraction must be in [0,1)")
    # Response rows deliberately remain unassigned until target/context splits exist.
    pools = np.full(len(cell_ids), "response_unassigned", dtype=object)
    pools[totals == 0] = "excluded_zero_library"
    for context in sorted(set(contexts)):
        rows = np.flatnonzero(is_control & (contexts == context) & (totals > 0))
        if kind == "challenge_controls":
            pools[rows] = "inference"
            continue
        # Sort by stable cell identifier before shuffling, independent of source row order.
        rows = rows[np.argsort(np.asarray(cell_ids)[rows])]
        rows = stable_rng(seed, f"{source_id}:{context}:controls").permutation(rows)
        n_reference = int(np.floor(reference_fraction * len(rows)))
        if reference_fraction > 0 and len(rows) >= 2:
            n_reference = max(1, n_reference)
        pools[rows[:n_reference]] = "evaluation_reference"
        pools[rows[n_reference:]] = "fit"
    return pools


@dataclass
class ObservationBatch:
    x: np.ndarray
    measurement_mask: np.ndarray
    output_x: np.ndarray
    output_measurement_mask: np.ndarray
    raw_counts: sparse.csr_matrix
    library_totals: np.ndarray
    cell_ids: np.ndarray
    targets: np.ndarray
    contexts: np.ndarray
    batches: np.ndarray
    is_control: np.ndarray
    rows: np.ndarray
    raw_output_counts: sparse.csr_matrix | None = None

    @property
    def encoder_input(self):
        return np.concatenate([self.x, self.measurement_mask.astype(np.float32)], axis=1)


class ObservationReader:
    """Densify only explicitly bounded minibatches, never entire source files."""

    def __init__(self, directory: str | Path, max_batch_rows: int = config.max_batch_rows, *,
                 cache_bytes: int = config.observation_cache_bytes):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "source.json").read_text())
        with (self.directory / "cells.csv").open(newline="") as handle:
            self.cells = list(csv.DictReader(handle))
        if max_batch_rows < 1:
            raise ValueError("max_batch_rows must be positive")
        self.max_batch_rows = max_batch_rows
        self.genes = self.manifest["source_genes"]
        self.encoder_genes = self.manifest["encoder_genes"]
        self.output_genes = self.manifest["output_genes"]
        self._shards = self.manifest["shards"]
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self.cache_bytes = cache_bytes
        self._cache = OrderedDict()
        self._cache_size = self._cache_hits = self._cache_misses = 0
        self._cache_lock = RLock()
        self._normalization_target = config.normalization_target_sum
        self._encoder_projection = _GeneProjection(self.genes, self.encoder_genes)
        self._output_projection = _GeneProjection(self.genes, self.output_genes)
        self._columns = {name: np.asarray([cell[name] for cell in self.cells])
                         for name in ("cell_id", "target", "context", "batch", "is_control", "pool")}
        self._stops = np.asarray([s["stop"] for s in self._shards], dtype=np.int64)
        starts = np.asarray([s["start"] for s in self._shards], dtype=np.int64)
        if (not len(starts) or starts[0] != 0 or np.any(self._stops <= starts)
                or not np.array_equal(starts[1:], self._stops[:-1]) or self._stops[-1] != len(self.cells)):
            raise ValueError("Shard index must cover the cell axis contiguously and in order")
        self._control_rows = {}
        self._response_rows = {}
        for row, cell in enumerate(self.cells):
            if cell["is_control"] == "1":
                self._control_rows.setdefault((cell["context"], cell["pool"]), []).append(row)
            elif cell["pool"] == "response_unassigned":
                self._response_rows.setdefault((cell["context"], cell["target"]), []).append(row)

    def _load_shard(self, index):
        # A single reader may be used by prefetch and evaluation; cache accounting
        # and loads are serialized. Returned sparse blocks are internal/read-only.
        with self._cache_lock:
            if config.normalization_target_sum != self._normalization_target:
                raise ValueError("Create a new reader after changing normalization_target_sum")
            if index in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(index)
                return self._cache[index][0]
            self._cache_misses += 1
            raw = sparse.load_npz(self.directory / self._shards[index]["raw"]).tocsr()
            raw_bytes = raw.data.nbytes + raw.indices.nbytes + raw.indptr.nbytes
            normalized_bytes = raw.nnz * (np.dtype(np.float32).itemsize + raw.indices.itemsize) + raw.indptr.nbytes
            if raw_bytes + normalized_bytes + raw.shape[0] * 8 > self.cache_bytes:
                # With retention disabled or an oversized shard, normalize only
                # requested rows rather than doing uncachable full-shard work.
                return raw, None, None
            normalized, totals = log_cp10k(raw)
            entry = (raw, normalized, totals)
            size = sum(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes for a in (raw, normalized)) + totals.nbytes
            if size <= self.cache_bytes:
                while self._cache and self._cache_size + size > self.cache_bytes:
                    _, (_, old_size) = self._cache.popitem(last=False)
                    self._cache_size -= old_size
                self._cache[index] = (entry, size)
                self._cache_size += size
            return entry

    def cache_info(self):
        with self._cache_lock:
            return {"hits": self._cache_hits, "misses": self._cache_misses,
                    "bytes": self._cache_size, "limit_bytes": self.cache_bytes, "shards": len(self._cache)}

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
            self._cache_size = 0

    def rows(self, indices) -> ObservationBatch:
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not 0 < len(indices) <= self.max_batch_rows:
            raise ValueError("Request a nonempty integer row vector within max_batch_rows")
        if np.any(indices < 0) or np.any(indices >= len(self.cells)):
            raise ValueError("Row index out of bounds")
        pieces, normalized_pieces, total_pieces, positions = [], [], [], []
        shard_ids = np.searchsorted(self._stops, indices, side="right")
        for shard_id in np.unique(shard_ids):
            hits = np.flatnonzero(shard_ids == shard_id)
            local = indices[hits] - self._shards[shard_id]["start"]
            block, normalized, totals = self._load_shard(int(shard_id))
            selected = block[local]
            pieces.append(selected)
            if normalized is None:
                selected_normalized, selected_totals = log_cp10k(selected)
            else:
                selected_normalized, selected_totals = normalized[local], totals[local]
            normalized_pieces.append(selected_normalized)
            total_pieces.append(selected_totals)
            positions.extend(hits.tolist())
        order = np.argsort(positions)
        raw = sparse.vstack(pieces, format="csr")[order]
        normalized = sparse.vstack(normalized_pieces, format="csr")[order]
        totals = np.concatenate(total_pieces)[order]
        # Full-source totals were computed before either cached gene projection.
        x, output_x = self._encoder_projection(normalized), self._output_projection(normalized)
        mask, output_mask = self._encoder_projection.mask, self._output_projection.mask
        raw_output = self._output_projection(raw)
        col = lambda name: self._columns[name][indices].copy()
        return ObservationBatch(
            x.toarray(), np.broadcast_to(mask, x.shape).copy(),
            output_x.toarray(), np.broadcast_to(output_mask, output_x.shape).copy(),
            raw, totals, col("cell_id"), col("target"), col("context"), col("batch"),
            col("is_control") == "1", indices.copy(), raw_output,
        )

    def sample_controls(self, *, context, pool, n=config.cells_per_group, seed=config.seed):
        if pool not in ("fit", "evaluation_reference", "inference"):
            raise ValueError("Choose an explicit control pool")
        eligible = self._control_rows.get((context, pool), [])
        if not eligible:
            raise ValueError(f"No {pool} controls in context {context}")
        if not 0 < n <= self.max_batch_rows:
            raise ValueError("n exceeds minibatch limit")
        rows = np.random.default_rng(seed).choice(eligible, n, replace=len(eligible) < n)
        return self.rows(rows)

    def sample_group(self, *, target, context, permitted_response_rows,
                     control_reader, verified_context_map=None, transfer_context=None,
                     n=config.cells_per_group, seed=config.seed):
        """Sample responses and VCC 2026 controls independently.

        Cross-dataset context correspondence must be supplied explicitly after
        verification. Batch identifiers from different datasets are not assumed to
        identify a common experimental batch. No source-control fallback is used.
        Response rows must come from the caller's frozen-split allowlist.
        Alternatively, transfer_context explicitly selects an anonymous 2026
        context for experimental cross-context transfer, without asserting a match.
        """
        if self.manifest["kind"] != "supervised":
            raise ValueError("Challenge controls cannot supply supervised response groups")
        if (control_reader.manifest["kind"] != "challenge_controls"
                or control_reader.manifest.get("dataset") != "vcc2026"):
            raise ValueError("Unperturbed observations must come from VCC 2026 controls")
        if (self.encoder_genes != control_reader.encoder_genes
                or self.output_genes != control_reader.output_genes
                or self.manifest["gene_identifier_namespace"] != control_reader.manifest["gene_identifier_namespace"]):
            raise ValueError("Perturbed and unperturbed readers must share verified gene axes and namespace")
        if transfer_context is not None:
            if verified_context_map is not None:
                raise ValueError("Choose explicit cross-context transfer or a verified mapping, not both")
            control_context = str(transfer_context)
        else:
            if not isinstance(verified_context_map, dict) or context not in verified_context_map:
                raise ValueError("Supply a verified context mapping or an explicit experimental transfer_context")
            control_context = verified_context_map[context]
        if control_context not in control_reader.manifest["contexts"]:
            raise ValueError("Mapped VCC 2026 context is absent from the selected control reader")
        allowed = np.asarray(permitted_response_rows)
        if (allowed.ndim != 1 or allowed.dtype.kind not in "iu" or np.any(allowed < 0)
                or np.any(allowed >= len(self.cells))):
            raise ValueError("Invalid frozen-split response row allowlist")
        if not 0 < n <= min(self.max_batch_rows, control_reader.max_batch_rows):
            raise ValueError("n exceeds minibatch limit")
        responses = np.intersect1d(allowed, self._response_rows.get((context, target), []))
        controls = control_reader._control_rows.get((control_context, "inference"), [])
        if not len(responses) or not controls:
            raise ValueError("No eligible responses or VCC 2026 controls for this group")
        response_rng = stable_rng(seed, "responses")
        control_rng = stable_rng(seed, "controls")
        response_rows = response_rng.choice(responses, n, replace=len(responses) < n)
        control_rows = control_rng.choice(controls, n, replace=len(controls) < n)
        return {"perturbed": self.rows(response_rows), "unperturbed": control_reader.rows(control_rows),
                "response_context": context, "control_context": control_context,
                "output_genes": list(self.output_genes),
                "cross_context_transfer": transfer_context is not None,
                "context_relationship": "experimental_transfer" if transfer_context is not None else "verified_mapping",
                "unperturbed_dataset": "vcc2026", "batch_matched": False, "paired_cells": False}
