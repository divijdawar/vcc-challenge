"""Explicit, read-only H5AD adapters. Read expression by row blocks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def strings(node) -> np.ndarray:
    if isinstance(node, h5py.Group):
        encoding = text(node.attrs.get("encoding-type", ""))
        if encoding == "categorical":
            categories = strings(node["categories"])
            codes = node["codes"][:]
            if np.any(codes < 0) or np.any(codes >= len(categories)):
                raise ValueError(f"Missing/invalid categorical metadata: {node.name}")
            return categories[codes]
        if "values" in node:
            if "mask" in node and np.any(node["mask"][:]):
                raise ValueError(f"Missing metadata: {node.name}")
            return strings(node["values"])
        raise ValueError(f"Unsupported metadata encoding at {node.name}: {encoding}")
    values = node[:]
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional identifiers at {node.name}")
    result = np.asarray([text(v) for v in values], dtype=str)
    if any(not v.strip() for v in result):
        raise ValueError(f"Empty metadata at {node.name}")
    return result


def unique_ids(values, name: str) -> list[str]:
    result = [text(v) for v in values]
    if not result or any(not v.strip() for v in result):
        raise ValueError(f"{name} must contain nonempty identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate {name}; resolve aliases/duplicates explicitly")
    return result


class CountSource:
    def __init__(self, config: dict):
        self.config = config
        if config.get("counts_are_raw") is not True:
            raise ValueError("Explicit counts_are_raw: true declaration is required")
        self.handle = h5py.File(config["path"], "r")
        try:
            location = config["counts_location"]
            if location != "X" and location != "raw/X" and not location.startswith("layers/"):
                raise ValueError("counts_location must be X, raw/X, or layers/NAME")
            self.matrix = self.handle[location]
            if isinstance(self.matrix, h5py.Dataset):
                self.shape = self.matrix.shape
                self.storage = "dense"
            else:
                self.storage = text(self.matrix.attrs.get("encoding-type", ""))
                if self.storage != "csr_matrix":
                    raise ValueError("Sparse counts must use CSR storage for bounded row reads")
                self.shape = tuple(int(v) for v in self.matrix.attrs["shape"])
                if (self.matrix["indptr"].shape != (self.shape[0] + 1,)
                        or self.matrix["data"].shape != self.matrix["indices"].shape
                        or self.matrix["indptr"][0] != 0
                        or self.matrix["indptr"][-1] != len(self.matrix["data"])):
                    raise ValueError("Invalid CSR dimensions/indptr")
            if len(self.shape) != 2 or not all(self.shape):
                raise ValueError("Counts must be a nonempty two-dimensional matrix")
            var = self.handle["raw/var" if location == "raw/X" else "var"]
            gene_key = config.get("gene_key", "_index")
            if gene_key == "_index":
                gene_key = text(var.attrs.get("_index", "_index"))
            self.original_genes = unique_ids(strings(var[gene_key]), "gene identifiers")
            aliases = config.get("gene_aliases", {})
            self.genes = unique_ids([aliases.get(g, g) for g in self.original_genes], "canonical gene identifiers")
            obs = self.handle["obs"]
            index_key = text(obs.attrs.get("_index", "_index"))
            self.cell_ids = np.asarray(unique_ids(strings(obs[index_key]), "cell identifiers"))
            self.labels = strings(obs[config["target_key"]])
            if "context_key" in config:
                self.contexts = strings(obs[config["context_key"]])
            elif config.get("context"):
                self.contexts = np.full(self.shape[0], str(config["context"]))
            else:
                raise ValueError("Specify context_key or an explicit context")
            self.batches = (strings(obs[config["batch_key"]]) if config.get("batch_key")
                            else np.full(self.shape[0], ""))
            if len(self.genes) != self.shape[1] or any(
                len(v) != self.shape[0] for v in [self.cell_ids, self.labels, self.contexts, self.batches]
            ):
                raise ValueError("Expression and metadata dimensions disagree")
            controls = config["control_labels"]
            if not isinstance(controls, list) or not controls or any(not isinstance(v, str) or not v for v in controls):
                raise ValueError("control_labels must be an explicit nonempty list of strings")
            self.is_control = np.isin(self.labels, controls)
            label_map = config.get("target_aliases", {})
            self.targets = np.asarray(["" if c else label_map.get(t, t)
                                       for t, c in zip(self.labels, self.is_control)], dtype=str)
            if any(not v.strip() for v in self.targets[~self.is_control]):
                raise ValueError("Perturbed cells require explicit target identifiers")
            self.kind = config["kind"]
            if self.kind not in ("supervised", "challenge_controls"):
                raise ValueError("kind must be supervised or challenge_controls")
            if self.kind == "challenge_controls" and not self.is_control.all():
                raise ValueError("Challenge-control input contains perturbed/unrecognized labels")
        except Exception:
            self.close()
            raise

    def close(self):
        self.handle.close()

    def rows(self, start: int, stop: int) -> sparse.csr_matrix:
        if not 0 <= start < stop <= self.shape[0]:
            raise ValueError("Invalid row interval")
        if self.storage == "dense":
            block = sparse.csr_matrix(self.matrix[start:stop, :])
        else:
            pointers = self.matrix["indptr"][start:stop + 1]
            lo, hi = int(pointers[0]), int(pointers[-1])
            if np.any(np.diff(pointers) < 0):
                raise ValueError("CSR indptr is not monotonic")
            indices = self.matrix["indices"][lo:hi]
            if np.any(indices < 0) or np.any(indices >= self.shape[1]):
                raise ValueError("CSR column index out of range")
            block = sparse.csr_matrix((self.matrix["data"][lo:hi], indices, pointers - lo),
                                      shape=(stop - start, self.shape[1]))
        values = block.data
        if (not np.isfinite(values).all() or np.any(values < 0)
                or not np.equal(values, np.rint(values)).all()):
            raise ValueError(f"{self.config['id']}: counts must be finite, nonnegative integers; check count location")
        # Preserve the source data dtype in the raw cache. Normalize a separate copy.
        return block
