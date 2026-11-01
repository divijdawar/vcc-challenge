"""Bounded-memory sparse H5AD writing and VCC upload-contract validation."""
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy import sparse

from .config import config
from .h5ad import strings, unique_ids


@dataclass(frozen=True)
class SubmissionContract:
    genes: tuple[str, ...]
    contexts: tuple[str, ...]
    targets: tuple[str, ...]
    cells_per_target: int = config.cells_per_target
    control_label: str = config.control_label
    max_counts_per_cell: int = config.max_counts_per_cell
    max_cells: int = config.max_submission_cells
    max_entries: int = config.max_submission_entries

    def __post_init__(self):
        for name in ("genes", "contexts", "targets"):
            unique_ids(getattr(self, name), name)
        if self.control_label in self.targets or self.cells_per_target < 1:
            raise ValueError("Submission target panel cannot contain the control condition")
        if self.expected_rows > self.max_cells:
            raise ValueError("Required panel exceeds the submission cell cap")

    @property
    def expected_rows(self):
        return len(self.contexts) * len(self.targets) * self.cells_per_target


def legal_counts(counts, max_counts_per_cell):
    raw = sparse.csr_matrix(counts)
    if (not np.isfinite(raw.data).all() or np.any(raw.data < 0)
            or not np.equal(raw.data, np.rint(raw.data)).all()):
        raise ValueError("Submission counts must be finite, nonnegative integers")
    if np.any(raw.data > max_counts_per_cell):
        raise ValueError("A single entry exceeds the per-cell count limit")
    # Aggregate sparse duplicates in wide arithmetic before checking row totals.
    raw = raw.astype(np.int64)
    raw.sum_duplicates()
    raw.eliminate_zeros()
    raw.sort_indices()
    if np.any(np.asarray(raw.sum(axis=1)).ravel() > max_counts_per_cell):
        raise ValueError("Submission has an excessive per-cell library total")
    return raw.astype(np.int32)


def _string_dataset(group, name, values=None):
    if values is None:
        dataset = group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=True,
                                       dtype=h5py.string_dtype("utf-8"))
    else:
        dataset = group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))
    dataset.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})
    return dataset


class SubmissionWriter:
    """Writes an atomic predictions-only H5AD; controls stay in their original files."""
    def __init__(self, path, contract: SubmissionContract):
        self.path, self.contract = Path(path), contract
        if self.path.exists():
            raise ValueError(f"Refusing to overwrite submission: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}-", suffix=".h5ad", dir=self.path.parent)
        os.close(fd)
        self.temporary = Path(temporary)
        self.handle = h5py.File(self.temporary, "w")
        self.handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        self.x = self.handle.create_group("X")
        self.x.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": (0, len(contract.genes))})
        for name, dtype in [("data", np.int32), ("indices", np.int32), ("indptr", np.int64)]:
            self.x.create_dataset(name, data=np.asarray([0] if name == "indptr" else [], dtype=dtype),
                                  maxshape=(None,), chunks=True, compression="gzip")
        obs = self.handle.create_group("obs")
        self.obs_columns = ("target_gene", "context", "source_cell_id")
        obs.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index",
                          "column-order": np.asarray(self.obs_columns, dtype=h5py.string_dtype())})
        self.obs = {name: _string_dataset(obs, name) for name in ("_index", *self.obs_columns)}
        var = self.handle.create_group("var")
        var.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index",
                          "column-order": np.asarray([], dtype="S1")})
        _string_dataset(var, "_index", contract.genes)
        for name in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            group = self.handle.create_group(name)
            group.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
        self.counts, self.rows, self.entries = Counter(), 0, 0

    def append(self, counts, *, context, target, row_ids, source_cell_ids):
        if context not in self.contract.contexts or target not in self.contract.targets or target == self.contract.control_label:
            raise ValueError("Unknown context/target or forbidden control row in submission")
        matrix = legal_counts(counts, self.contract.max_counts_per_cell)
        rows, genes = matrix.shape
        if genes != len(self.contract.genes) or len(row_ids) != rows or len(source_cell_ids) != rows:
            raise ValueError("Submission block axes/metadata disagree")
        if (self.rows + rows > self.contract.max_cells or self.entries + matrix.nnz > self.contract.max_entries
                or self.counts[context, target] + rows > self.contract.cells_per_target):
            raise ValueError("Submission exceeds cell, stored-entry, or target-group count limits")
        for name, values in [("data", matrix.data), ("indices", matrix.indices),
                             ("indptr", matrix.indptr[1:] + self.entries)]:
            dataset = self.x[name]
            start = len(dataset)
            dataset.resize((start + len(values),))
            dataset[start:] = values
        values = {"_index": row_ids, "target_gene": [target] * rows, "context": [context] * rows,
                  "source_cell_id": source_cell_ids}
        for name, data in values.items():
            self.obs[name].resize((self.rows + rows,))
            self.obs[name][self.rows:] = data
        self.rows += rows
        self.entries += matrix.nnz
        self.counts[context, target] += rows
        self.x.attrs["shape"] = (self.rows, genes)

    def finish(self):
        expected = {(context, target): self.contract.cells_per_target
                    for context in self.contract.contexts for target in self.contract.targets}
        if self.rows != self.contract.expected_rows or dict(self.counts) != expected:
            raise ValueError("Submission is missing required target/context cells")
        self.handle.close()
        report = validate_submission(self.temporary, self.contract)
        if self.path.exists():
            raise ValueError("Submission destination appeared during generation")
        self.temporary.rename(self.path)
        return report

    def abort(self):
        self.handle.close()
        self.temporary.unlink(missing_ok=True)


def validate_submission(path, contract: SubmissionContract):
    """Validate locally in sparse chunks; official vcc prep is still required."""
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("Submission must use sparse CSR storage")
        shape = tuple(x.attrs["shape"])
        if shape != (contract.expected_rows, len(contract.genes)):
            raise ValueError("Submission matrix dimensions do not match the release")
        genes = strings(handle["var"][handle["var"].attrs["_index"]]).tolist()
        if genes != list(contract.genes):
            raise ValueError("Submission gene order differs from the release")
        obs = handle["obs"]
        identifiers = unique_ids(strings(obs[obs.attrs["_index"]]), "submission row IDs")
        contexts, targets = strings(obs["context"]), strings(obs["target_gene"])
        if len(identifiers) != shape[0] or len(contexts) != shape[0] or len(targets) != shape[0]:
            raise ValueError("Submission metadata row counts disagree")
        expected = {(context, target): contract.cells_per_target for context in contract.contexts for target in contract.targets}
        if dict(Counter(zip(contexts, targets))) != expected or contract.control_label in targets:
            raise ValueError("Submission must contain exactly the target panel and no control cells")
        entries = len(x["data"])
        if (shape[0] > contract.max_cells or entries > contract.max_entries
                or len(x["indices"]) != entries or len(x["indptr"]) != shape[0] + 1
                or x["indptr"][0] != 0 or x["indptr"][-1] != entries):
            raise ValueError("Invalid CSR dimensions or submission caps exceeded")
        for start in range(0, shape[0], 256):
            stop = min(start + 256, shape[0])
            pointers = x["indptr"][start:stop + 1]
            if np.any(np.diff(pointers) < 0):
                raise ValueError("Invalid CSR row pointers")
            lo, hi = int(pointers[0]), int(pointers[-1])
            values, indices = x["data"][lo:hi], x["indices"][lo:hi]
            if np.any(indices < 0) or np.any(indices >= shape[1]):
                raise ValueError("Submission column index out of bounds")
            legal_counts(sparse.csr_matrix((values, indices, pointers - lo), shape=(stop - start, shape[1])),
                         contract.max_counts_per_cell)
    return {"valid": True, "rows": int(shape[0]), "genes": int(shape[1]), "stored_entries": entries,
            "controls_included": False, "official_cli_validated": False}
