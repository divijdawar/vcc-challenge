"""Local VCC 2025 response -> VCC 2026 control training orchestration.

No downloads, submission, or training happens on import. Split selection uses
metadata only; gradient/reference fitting reads only frozen training populations.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .config import config, __file__ as config_file
from .h5ad import CountSource, sha256
from .observations import ObservationReader, stable_rng
from .prepare import resolve_data_inputs, validate_prepared
from .splits import freeze_splits, load_splits, split_summary


def project_path(value):
    root = (Path(config_file).resolve().parent / config.project_root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Run, split, and prepared-cache paths must remain inside the project')
    return path


def _source_adapter():
    supervised = [s for s in config.sources if s.get('kind') == 'supervised']
    if (config.perturbed_dataset != 'vcc2025' or config.unperturbed_dataset != 'vcc2026'
            or len(supervised) != 1 or supervised[0]['id'] != config.response_source_id
            or supervised[0].get('dataset') != 'vcc2025' or supervised[0].get('release_split') != 'train'):
        raise ValueError('This runner requires exactly one VCC 2025 training source and VCC 2026 model controls')
    return supervised[0]


def inspect_data():
    inputs = resolve_data_inputs()
    sources = []
    for adapter in inputs['sources']:
        source = CountSource(adapter)
        try:
            # Inspect bounded raw counts and explicit metadata; fit no features.
            source.rows(0, min(source.shape[0], config.chunk_rows))
            sources.append({'id': adapter['id'], 'dataset': adapter.get('dataset'),
                            'path': adapter['path'], 'shape': list(source.shape), 'storage': source.storage,
                            'controls': int(source.is_control.sum()), 'responses': int((~source.is_control).sum()),
                            'contexts': sorted(set(source.contexts)),
                            'output_gene_overlap': len(set(source.genes) & set(inputs['output_genes']))})
        finally:
            source.close()
    return {'sources': sources, 'scope': 'metadata and bounded raw-count sample; not a full count audit'}


def _readers():
    _source_adapter()
    directory = project_path(config.output_dir)
    report = json.loads((directory / 'report.json').read_text())
    summaries = {s['id']: s for s in report['sources']}
    expected_inputs = resolve_data_inputs()
    expected_adapters = {s['id']: s for s in expected_inputs['sources']}
    if set(summaries) != set(expected_adapters):
        raise ValueError('Prepared sources differ from config.sources/the selected 2026 release; prepare a new cache')
    resolved = json.loads((directory / 'resolved_config.json').read_text())
    if (resolved.get('release_manifest') != expected_inputs.get('release_manifest')
            or resolved['challenge_targets'] != expected_inputs['challenge_targets']):
        raise ValueError('Prepared release metadata differs from the selected release; prepare a new cache')
    readers = {}
    for source_id, summary in summaries.items():
        source_dir = directory / source_id
        if not source_dir.resolve().is_relative_to(directory):
            raise ValueError('Prepared source ID escaped the cache directory')
        if sha256(source_dir / 'source.json') != summary['manifest_sha256']:
            raise ValueError('Prepared source manifest hash mismatch')
        reader = ObservationReader(source_dir)
        if (reader.manifest['adapter'] != expected_adapters[source_id]
                or reader.encoder_genes != expected_inputs['encoder_genes']
                or reader.output_genes != expected_inputs['output_genes']
                or reader.manifest['normalization']['target_sum'] != config.normalization_target_sum):
            raise ValueError('Prepared adapter, gene axes, or normalization differ from config; prepare a new cache')
        if sha256(source_dir / 'cells.csv') != reader.manifest['cells_sha256']:
            raise ValueError('Prepared cell metadata hash mismatch')
        readers[source_id] = reader
    if config.response_source_id not in readers:
        raise ValueError('Prepared VCC 2025 source missing; run prepare with the new sources setting')
    response = readers.pop(config.response_source_id)
    if (response.manifest.get('dataset') != 'vcc2025' or response.manifest['kind'] != 'supervised'
            or response.manifest['adapter'].get('release_split') != 'train'):
        raise ValueError('Prepared response source is not the VCC 2025 training release')
    controls = {}
    for reader in readers.values():
        if reader.manifest['kind'] != 'challenge_controls' or reader.manifest.get('dataset') != 'vcc2026':
            raise ValueError('Unexpected supervised/non-2026 source in the prepared cache')
        if len(reader.manifest['contexts']) != 1:
            raise ValueError('Each 2026 reader must retain its release context identity')
        context = reader.manifest['contexts'][0]
        if context in controls:
            raise ValueError('Duplicate VCC 2026 control context')
        if response.encoder_genes != reader.encoder_genes or response.output_genes != reader.output_genes:
            raise ValueError('Response and control prepared gene axes differ')
        controls[context] = reader
    release = resolved.get('release_manifest', {})
    if not controls or set(controls) != set(map(str, release.get('contexts', []))):
        raise ValueError('Prepared 2026 contexts must match the selected release manifest exactly')
    return response, controls


def split_from_config():
    response, _ = _readers()
    manifest = freeze_splits(response, project_path(config.split_manifest), seed=config.seed,
                             test_fraction=config.test_target_fraction,
                             development_fraction=config.development_target_fraction,
                             cell_fraction=config.development_cell_fraction,
                             min_cells=config.min_response_cells, max_cells=config.max_response_cells_per_target)
    return {'split_manifest': str(project_path(config.split_manifest)), 'partitions': split_summary(manifest),
            'expression_fitted': False}


def training_readiness():
    """Read-only preflight: no tensor/model creation, output directories, or download."""
    adapter = _source_adapter()
    root = (Path(config_file).resolve().parent / config.project_root).resolve()
    response_path = (root / adapter['path']).resolve()
    missing = []
    if not response_path.is_file():
        missing.append(f'Place the official 2025 training H5AD at {response_path}, or edit config.sources[0]["path"]')
    report_path = project_path(config.output_dir) / 'report.json'
    if not report_path.is_file():
        missing.append('Run python -m vcc prepare after configuring the local input files')
    split_path = project_path(config.split_manifest)
    if not split_path.is_file():
        missing.append('Run python -m vcc split to freeze training/development/test membership')
    result = {'ready': not missing, 'perturbed_dataset': 'vcc2025', 'unperturbed_dataset': 'vcc2026',
              'response_path': str(response_path), 'prepared_cache': str(report_path.parent),
              'split_manifest': str(split_path), 'control_context_policy': config.control_context_policy,
              'experimental_transfer': config.control_context_policy == 'uniform_transfer', 'missing': missing,
              'downloads_performed': False, 'training_started': False}
    _validate_settings()
    project_path(config.run_directory)
    if report_path.is_file():
        response, controls = _readers()
        _validate_contexts(controls)
        result['control_contexts'] = sorted(controls)
        if config.cells_per_group > response.max_batch_rows:
            missing.append('Reduce cells_per_group to the prepared response reader minibatch bound')
        if not response._control_rows.get(('H1', 'fit')):
            missing.append('VCC 2025 needs nonzero fitting controls for its effect reference')
        for context, reader in controls.items():
            if not reader._control_rows.get((context, 'inference')):
                missing.append(f'VCC 2026 context {context} has no eligible nonzero controls')
        if sum(response.manifest['output_measurement_mask']) < 2:
            missing.append('At least two measured output genes are required for target-masked population training')
        if split_path.is_file():
            split = load_splits(response, split_path)
            result['partitions'] = split_summary(split)
            result['condition_dimension'] = len(split['train_targets']) + int(config.include_control_indicator)
            result['input_dimension'], result['output_dimension'] = len(response.encoder_genes), len(response.output_genes)
            if not any(split['groups']['development_seen'].values()):
                missing.append('Reserve seen-target development cells when freezing the split')
            release_targets = json.loads((report_path.parent / 'resolved_config.json').read_text())['challenge_targets']
            trained = set(split['train_targets'])
            result['target_coverage'] = {'release_targets': len(release_targets),
                                         'learned': sorted(set(release_targets) & trained),
                                         'no_change_fallback': sorted(set(release_targets) - trained)}
    result['ready'] = not missing
    return result


def _validate_settings():
    for name in ('training_epochs', 'groups_per_step', 'cells_per_group', 'log_interval', 'validation_interval',
                 'checkpoint_interval', 'validation_groups', 'population_max_controls'):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f'config.{name} must be a positive integer')
    if config.planned_steps is not None and (type(config.planned_steps) is not int or config.planned_steps < 1):
        raise ValueError('planned_steps must be None or a positive integer')
    if config.control_context_policy not in ('uniform_transfer', 'verified_mapping'):
        raise ValueError('Choose uniform_transfer or verified_mapping')
    if config.training_objective != 'residual':
        raise ValueError('This cross-dataset runner uses the residual count objective')
    if config.alignment not in ('mmd', 'none'):
        raise ValueError('Cross-dataset cells are unpaired; choose MMD or disabled alignment')
    if config.allow_unpaired_reference_alignment:
        raise ValueError('This runner does not permit claiming cross-dataset rows are paired')


def _validate_contexts(controls):
    if any(config.cells_per_group > reader.max_batch_rows for reader in controls.values()):
        raise ValueError('cells_per_group exceeds the control reader minibatch bound')
    if config.control_context_policy == 'verified_mapping':
        if config.verified_context_map.get('H1') not in controls:
            raise ValueError('Provide a verified H1 -> selected 2026 context mapping')


def _sample(response, controls, target, rows, seed, *, context=None):
    contexts = sorted(controls)
    if config.control_context_policy == 'uniform_transfer':
        context = context or str(stable_rng(seed, 'control_context').choice(contexts))
        kwargs = {'transfer_context': context}
    else:
        context = config.verified_context_map['H1']
        kwargs = {'verified_context_map': config.verified_context_map}
    return response.sample_group(target=target, context='H1', permitted_response_rows=np.asarray(rows, dtype=np.int64),
                                 control_reader=controls[context], n=config.cells_per_group, seed=seed, **kwargs)


def train_from_config():
    setup_start = time.perf_counter()
    readiness = training_readiness()
    if not readiness['ready']:
        raise ValueError('; '.join(readiness['missing']))
    if config.verify_cache_before_training:
        validate_prepared(project_path(config.output_dir))
    response, controls = _readers()
    split_path = project_path(config.split_manifest)
    split = load_splits(response, split_path)
    split_hash = sha256(split_path)
    validation_targets = [target for target, rows in split['groups']['development_seen'].items() if rows]
    train_rows = np.asarray([row for rows in split['groups']['train'].values() for row in rows], dtype=np.int64)
    if config.cells_per_group > response.max_batch_rows:
        raise ValueError('cells_per_group exceeds response reader minibatch bound')

    # Optional tinygrad dependency is loaded only for actual training.
    from tinygrad import Tensor, dtypes
    from tinygrad.device import Device
    from .batching import ModelBatch
    from .checkpoint import save_prediction_bundle
    from .conditioning import PerturbationVocabulary
    from .losses import LossWeights, PerturbedVAELoss, calibrate_mmd_bandwidth
    from .model import LatentNoise, ModelConfig, PerturbedVAE
    from .population import fit_population_reference
    from .training import RegularizationSchedule, Trainer

    Tensor.manual_seed(config.seed)
    vocabulary = PerturbationVocabulary(tuple(split['train_targets']), config.control_label, config.include_control_indicator)
    model_config = ModelConfig(
        input_dim=len(response.encoder_genes), output_dim=len(response.output_genes), condition_dim=vocabulary.dimension,
        hidden_dim=config.hidden_dim, invariant_dim=config.invariant_dim, responsive_dim=config.responsive_dim,
        n_layers=config.n_layers, negative_slope=config.negative_slope,
        append_measurement_mask=config.append_measurement_mask,
        sample_logvar_min=config.sample_logvar_min, sample_logvar_max=config.sample_logvar_max,
        consistent_logvar_clamp=config.consistent_logvar_clamp, zero_initialize_structure=config.zero_initialize_structure)
    model = PerturbedVAE(model_config, device=config.training_device)
    convert = lambda a: Tensor(np.asarray(a, dtype=np.float32), device=model.device, dtype=dtypes.float32)
    support = np.asarray(response.manifest['output_measurement_mask'], dtype=bool)
    if not support.any():
        raise ValueError('No decoder genes are measured in the gradient-training source')
    reference = fit_population_reference(response, 'H1', train_rows, split_manifest_hash=split_hash,
                                         device=model.device, max_controls=config.population_max_controls)
    bandwidth = config.mmd_bandwidth
    if config.alignment == 'mmd' and bandwidth is None:
        means = []
        for context, reader in sorted(controls.items()):
            calibration = reader.sample_controls(context=context, pool='inference', n=config.cells_per_group,
                                                   seed=config.seed)
            means.append(model.encode_invariant(convert(calibration.x), convert(calibration.measurement_mask)).mean.detach())
        combined = means[0].cat(*means[1:], dim=0) if len(means) > 1 else means[0]
        bandwidth = calibrate_mmd_bandwidth(combined)
    weights = LossWeights(config.reconstruction_weight, config.responsive_kl_weight, config.invariant_kl_weight,
                          config.alignment_weight, config.sparsity_weight, config.population_weight)
    objective = PerturbedVAELoss(weights, alignment=config.alignment, mmd_bandwidth=bandwidth,
                                mode=config.training_objective)
    planned = config.planned_steps or config.training_epochs * math.ceil(len(train_rows) / (config.cells_per_group * config.groups_per_step))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = project_path(config.run_directory) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(model, objective, RegularizationSchedule(planned, config.warmup_fraction, config.warmup_curve),
                      learning_rate=config.learning_rate, gradient_clip=config.gradient_clip,
                      jit=config.training_jit, jit_cache_size=config.jit_cache_size,
                      diagnostics_directory=directory / 'diagnostics')
    metadata = {
        'schema_version': 1, 'seed': config.seed, 'model': asdict(model_config), 'device': model.device,
        'software': {'python': platform.python_version(), 'tinygrad': version('tinygrad')},
        'configuration': {k: v for k, v in vars(config).items() if not k.startswith('_') and not callable(v)},
        'split_manifest': str(split_path), 'split_manifest_sha256': split_hash, 'split_summary': split_summary(split),
        'vocabulary': vocabulary.manifest(), 'encoder_genes': response.encoder_genes, 'output_genes': response.output_genes,
        'supported_output_mask': support.tolist(), 'mmd_bandwidth': bandwidth, 'planned_steps': planned,
        'source_manifest_sha256': sha256(response.directory / 'source.json'),
        'control_manifests_sha256': {c: sha256(r.directory / 'source.json') for c, r in controls.items()},
        'control_context_policy': config.control_context_policy, 'context_identities_inferred': False,
        'population_loss': 'context-relative effects' if config.control_context_policy == 'uniform_transfer' else 'verified-context absolute profiles',
        'population_reference': {**reference.provenance, 'scale_squared': reference.scale_squared,
                                 'control_profile': reference.control_profile.numpy().tolist()},
        'held_out_target_policy': 'Excluded from vocabulary and fitting; no-change export fallback for unseen targets',
        'release_target_coverage': readiness['target_coverage'],
        'checkpoint_type': 'prediction bundles; optimizer/RNG resume is not implemented',
        'setup_elapsed_seconds': time.perf_counter() - setup_start,
        'code_hashes': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    (directory / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (directory / 'split.json').write_text(json.dumps(split, indent=2, sort_keys=True) + '\n')

    def checkpoint(name):
        save_prediction_bundle(directory / name, model, vocabulary, response.encoder_genes, response.output_genes,
                               support.tolist(), optimizer_steps=trainer.completed_steps,
                               training_metadata={'split_manifest_sha256': split_hash, 'run_id': run_id,
                                                  'control_context_policy': config.control_context_policy})
        return str(directory / name)

    # Fixed seen-target development rows are excluded from fitting. Target-level
    # development/test rows are never selected for this command's losses or fits.
    validation_targets = stable_rng(config.validation_seed, 'validation_targets').permutation(validation_targets).tolist()[:config.validation_groups]

    def validate():
        losses = []
        for i, target in enumerate(validation_targets):
            context = sorted(controls)[i % len(controls)]
            observations = _sample(response, controls, target, split['groups']['development_seen'][target],
                                   config.validation_seed + i, context=context)
            batch = ModelBatch.from_observation_group(observations, vocabulary, model,
                                                       supported_output_mask=support, population_reference=reference)
            batch.prepare_training(population=objective.weights.population > 0)
            rng = np.random.default_rng(config.validation_seed + i)
            def noise():
                return LatentNoise(convert(rng.standard_normal((config.cells_per_group, model_config.invariant_dim))),
                                   convert(rng.standard_normal((config.cells_per_group, model_config.responsive_dim))))
            batch.noise, batch.control_noise, batch.prediction_noise = noise(), noise(), noise()
            forward = batch.forward(model, reconstruct_controls=True)
            prediction = batch.predict(model, validated=True) if objective.weights.population > 0 else None
            loss, _ = objective(forward, batch.observed_output, output_mask=batch.output_mask, regularization_scale=1.0,
                                control_observed=batch.observed_controls, control_output_mask=batch.control_output_mask,
                                prediction=prediction, response_counts=batch.raw_responses, population_reference=reference,
                                population_mask=batch.population_mask, validated_batch=True,
                                prediction_controls=batch.raw_controls if batch.cross_context_transfer else None)
            finite = loss.isfinite() & (prediction.expected.finite if prediction is not None else loss.isfinite())
            values = Tensor.stack(loss.detach(), finite.cast(dtypes.float32)).numpy()
            if values[1] != 1.0:
                raise FloatingPointError('Nonfinite development loss/prediction')
            losses.append(float(values[0]))
        return {'loss': float(np.mean(losses)), 'groups': len(losses), 'scope': 'seen-target development proxy; not official VCC score'}

    def observation_steps():
        for step in range(planned):
            rng = stable_rng(config.seed, f'training_step:{step}')
            yield [_sample(response, controls, str(rng.choice(vocabulary.targets)), train_rows,
                           int(rng.integers(0, 2**32))) for _ in range(config.groups_per_step)]

    start, best_loss, best_bundle = time.perf_counter(), float('inf'), None
    status = {'run_directory': str(directory), 'status': 'running', 'completed_steps': 0}
    status_path = directory / 'status.json'
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    iterator = trainer.iter_observations(observation_steps(), vocabulary, supported_output_mask=support,
                                         population_references={'H1': reference}, prefetch_steps=config.prefetch_steps)
    try:
        with (directory / 'metrics.jsonl').open('x') as log:
            for metrics in iterator:
                step = trainer.completed_steps
                if step == 1 or step % config.log_interval == 0 or step == planned:
                    Device[model.device].synchronize()
                    record = {'kind': 'training', **metrics, 'elapsed_seconds': time.perf_counter() - start}
                    log.write(json.dumps(record) + '\n'); log.flush()
                    print(json.dumps(record), flush=True)
                if step % config.validation_interval == 0 or step == planned:
                    development = validate()
                    record = {'kind': 'development', 'completed_steps': step, **development}
                    log.write(json.dumps(record) + '\n'); log.flush()
                    print(json.dumps(record), flush=True)
                    if development['loss'] < best_loss:
                        best_loss = development['loss']
                        best_bundle = checkpoint(f'best_step_{step:08d}')
                if step % config.checkpoint_interval == 0 and step < planned:
                    checkpoint(f'step_{step:08d}')
            Device[model.device].synchronize()
            final_bundle = checkpoint('final')
            status.update(status='complete', completed_steps=trainer.completed_steps, final_bundle=final_bundle,
                          best_bundle=best_bundle, best_development_loss=best_loss,
                          elapsed_seconds=time.perf_counter() - start,
                          total_elapsed_seconds=time.perf_counter() - setup_start)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      completed_steps=trainer.completed_steps, error=str(error))
        # Only interruption gets a last completed model snapshot; do not save a
        # possibly nonfinite failing model as a usable prediction checkpoint.
        if isinstance(error, KeyboardInterrupt) and trainer.completed_steps > 0:
            Device[model.device].synchronize()
            status['interrupted_bundle'] = checkpoint('interrupted')
        raise
    finally:
        iterator.close()
        status_path.write_text(json.dumps(status, indent=2) + '\n')
    return status
