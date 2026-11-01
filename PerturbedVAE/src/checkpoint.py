"""Self-contained fitted-model bundles for residual prediction, without PyTorch."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile

from tinygrad.nn.state import get_state_dict, safe_load, safe_save

from .conditioning import PerturbationVocabulary
from .config import config
from .h5ad import sha256, unique_ids
from .model import ModelConfig, PerturbedVAE


@dataclass
class PredictionBundle:
    model: PerturbedVAE
    vocabulary: PerturbationVocabulary
    encoder_genes: list[str]
    output_genes: list[str]
    supported_output_mask: list[bool]
    metadata: dict


def save_prediction_bundle(directory, model, vocabulary, encoder_genes, output_genes,
                           supported_output_mask, *, optimizer_steps: int, training_metadata=None):
    """Call after fitting; support/vocabulary must come from the frozen training manifests.

    This is an inference bundle, not an optimizer/RNG resume checkpoint.
    """
    directory = Path(directory)
    if directory.exists() or type(optimizer_steps) is not int or optimizer_steps < 1:
        raise ValueError("Use a new bundle directory and a positive completed optimizer-step count")
    encoder_genes = unique_ids(encoder_genes, "encoder genes")
    output_genes = unique_ids(output_genes, "output genes")
    support = list(supported_output_mask)
    if (len(encoder_genes) != model.config.input_dim or len(output_genes) != model.config.decoded_dim
            or len(support) != len(output_genes) or any(value not in (True, False, 0, 1) for value in support)
            or vocabulary.dimension != model.config.condition_dim):
        raise ValueError("Model, vocabulary, support mask, and gene axes are incompatible")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        safe_save(get_state_dict(model), str(temporary / "weights.safetensors"))
        metadata = {
            "schema_version": 1, "prediction_method": "residual_expected_counts_v1",
            "model": asdict(model.config), "vocabulary": vocabulary.manifest(),
            "encoder_genes": encoder_genes, "output_genes": output_genes,
            "supported_output_mask": [bool(value) for value in support], "optimizer_steps": optimizer_steps,
            "count_policy": {"normalization_target_sum": config.normalization_target_sum},
            "training_metadata": training_metadata or {},
            "weights_sha256": sha256(temporary / "weights.safetensors"),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.rename(directory)
    except Exception:
        shutil.rmtree(temporary)
        raise


def load_prediction_bundle(directory, *, device=None):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("prediction_method") != "residual_expected_counts_v1"
            or metadata.get("optimizer_steps", 0) < 1):
        raise ValueError("Expected a fitted residual prediction bundle")
    if metadata["count_policy"]["normalization_target_sum"] != config.normalization_target_sum:
        raise ValueError("Runtime normalization differs from the fitted bundle's count policy")
    if sha256(directory / "weights.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Prediction weights hash mismatch")
    model = PerturbedVAE(ModelConfig(**metadata["model"]), device=device)
    manifest = metadata["vocabulary"]
    vocabulary = PerturbationVocabulary(tuple(manifest["targets"]), manifest["control_label"],
                                        manifest["include_control_indicator"])
    encoder = unique_ids(metadata["encoder_genes"], "checkpoint encoder genes")
    output = unique_ids(metadata["output_genes"], "checkpoint output genes")
    support = metadata["supported_output_mask"]
    if (len(encoder) != model.config.input_dim or len(output) != model.config.decoded_dim
            or len(support) != len(output) or any(type(value) is not bool for value in support)
            or vocabulary.dimension != model.config.condition_dim):
        raise ValueError("Prediction bundle axes, vocabulary, or support mask are inconsistent")
    state = get_state_dict(model)
    loaded = safe_load(str(directory / "weights.safetensors"))
    if set(state) != set(loaded):
        raise ValueError("Checkpoint parameter names do not match the model")
    for name, parameter in state.items():
        value = loaded[name].to(model.device)
        if value.shape != parameter.shape or not bool(value.isfinite().all().item()):
            raise ValueError(f"Invalid checkpoint parameter: {name}")
        parameter.assign(value).realize()
    return PredictionBundle(model, vocabulary, encoder, output, support, metadata)
