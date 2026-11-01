import argparse
import json
import sys

from .config import config
from .prepare import prepare, resolve_data_inputs, validate_prepared


def main():
    parser = argparse.ArgumentParser(description="Prepare, split, train, or predict with PerturbedVAE")
    parser.add_argument("command", choices=["inspect-data", "prepare", "validate", "split", "train", "model-info", "predict", "validate-submission"])
    parser.add_argument("--dry-run", action="store_true", help="Read-only training readiness check (train only)")
    args = parser.parse_args()
    if args.dry_run and args.command != "train":
        parser.error("--dry-run is supported only with train")
    try:
        if args.command in ("inspect-data", "split", "train"):
            from .experiment import inspect_data, split_from_config, train_from_config, training_readiness
            if args.command == "inspect-data":
                result = inspect_data()
            elif args.command == "split":
                result = split_from_config()
            else:
                result = training_readiness() if args.dry_run else train_from_config()
            print(json.dumps(result, indent=2))
            return int(args.dry_run and not result["ready"])
        if args.command == "model-info":
            from dataclasses import asdict
            from .model import ModelConfig
            from .losses import LossWeights
            model = ModelConfig(input_dim=config.input_dim, condition_dim=config.condition_dim,
                                output_dim=config.output_dim)
            print(json.dumps({
                "model": asdict(model),
                "datasets": {"perturbed": config.perturbed_dataset, "unperturbed": config.unperturbed_dataset,
                             "control_context_policy": config.control_context_policy,
                             "dimensions_for_training": "derived from prepared gene axes and frozen training vocabulary"},
                "loss": {"mode": config.training_objective, "alignment": config.alignment, "mmd_bandwidth": config.mmd_bandwidth,
                         "weights": asdict(LossWeights())},
                "training": {
                    "learning_rate": config.learning_rate, "gradient_clip": config.gradient_clip,
                    "warmup_fraction": config.warmup_fraction, "warmup_curve": config.warmup_curve,
                    "planned_steps": config.planned_steps,
                    "allow_unpaired_reference_alignment": config.allow_unpaired_reference_alignment,
                    "diagnostics_directory": config.diagnostics_directory,
                    "jit": config.training_jit, "jit_cache_size": config.jit_cache_size,
                    "observation_cache_bytes": config.observation_cache_bytes,
                    "prefetch_steps": config.prefetch_steps,
                    "groups_per_step": config.groups_per_step, "epochs": config.training_epochs,
                    "split_manifest": config.split_manifest,
                },
                "framework": "tinygrad", "weights_initialized": False, "training_started": False,
            }, indent=2))
            return 0
        if args.command in ("predict", "validate-submission"):
            from .prediction import predict_from_config, validate_from_config
            result = predict_from_config() if args.command == "predict" else validate_from_config()
            print(json.dumps(result, indent=2))
            return 0
        inputs = resolve_data_inputs()
        result = prepare(inputs) if args.command == "prepare" else validate_prepared(inputs["output_dir"])
    except (ValueError, KeyError, OSError, TypeError, ImportError, FloatingPointError, RuntimeError) as error:
        print(f"vcc: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("vcc: training interrupted; see the run status and any completed prediction checkpoint", file=sys.stderr)
        return 130
    if args.command == "prepare":
        print(json.dumps({"output_dir": inputs["output_dir"], "sources": result["sources"],
                          "gradient_training_ready": result["gradient_training_ready"]}, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
