"""Explicit reference one-hot/multi-hot perturbation labels, without random unknowns."""
from dataclasses import dataclass

from tinygrad import Tensor, dtypes

from .config import config


@dataclass(frozen=True)
class PerturbationVocabulary:
    targets: tuple[str, ...]
    control_label: str = config.control_label
    include_control_indicator: bool = config.include_control_indicator

    def __post_init__(self):
        if (not self.targets or len(set(self.targets)) != len(self.targets)
                or any(not isinstance(t, str) or not t.strip() for t in self.targets)
                or not self.control_label or self.control_label in self.targets):
            raise ValueError("Use unique nonempty training target identifiers, excluding the control label")

    @property
    def dimension(self):
        return len(self.targets) + int(self.include_control_indicator)

    def encode(self, labels, *, device=None):
        """A string is one target/control; a tuple/list explicitly names a combination.

        Control is zero on target coordinates. An optional appended control bit is
        a VCC extension. Unknown targets raise: this reference core cannot establish
        unseen-gene generalization from untrained condition coordinates.
        """
        if isinstance(labels, str):
            raise ValueError("Supply a sequence of cell conditions, not a single string")
        positions = {target: i for i, target in enumerate(self.targets)}
        rows = []
        for label in labels:
            row = [0.0] * self.dimension
            if isinstance(label, str) and label == self.control_label:
                if self.include_control_indicator:
                    row[-1] = 1.0
            else:
                members = (label,) if isinstance(label, str) else tuple(label)
                if not members or len(set(members)) != len(members):
                    raise ValueError("A perturbation combination must contain unique target identifiers")
                for target in members:
                    if target not in positions:
                        raise ValueError(f"Unknown perturbation target: {target}")
                    row[positions[target]] = 1.0
            rows.append(row)
        if not rows:
            raise ValueError("Condition batch cannot be empty")
        return Tensor(rows, device=device, dtype=dtypes.float32)

    def manifest(self):
        return {"targets": list(self.targets), "control_label": self.control_label,
                "include_control_indicator": self.include_control_indicator,
                "dimension": self.dimension, "unknown_policy": "error"}
