"""Frozen target/cell splits of the public VCC 2025 TRAINING release only."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .config import config
from .h5ad import sha256
from .observations import stable_rng


PARTITIONS = ('train', 'development_seen', 'development_unseen', 'test')


def _source_fingerprint(reader):
    if (reader.manifest.get('dataset') != 'vcc2025' or reader.manifest['kind'] != 'supervised'
            or reader.manifest['adapter'].get('release_split') != 'train'):
        raise ValueError('Split only the official VCC 2025 training release, never its validation/test files')
    if set(reader.manifest['contexts']) != {'H1'}:
        raise ValueError('VCC 2025 response source must have the explicit H1 context')
    cells_hash = sha256(reader.directory / 'cells.csv')
    if cells_hash != reader.manifest['cells_sha256']:
        raise ValueError('Prepared source cell metadata hash mismatch')
    return {'id': reader.manifest['id'], 'manifest_sha256': sha256(reader.directory / 'source.json'),
            'cells_sha256': cells_hash}


def freeze_splits(reader, path, *, seed=config.seed, test_fraction=config.test_target_fraction,
                  development_fraction=config.development_target_fraction,
                  cell_fraction=config.development_cell_fraction, min_cells=config.min_response_cells,
                  max_cells=config.max_response_cells_per_target):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Split manifest already exists; use it or choose a new path: {path}')
    if (any(not math.isfinite(x) or not 0 <= x < 1 for x in (test_fraction, development_fraction, cell_fraction))
            or test_fraction + development_fraction >= 1 or type(min_cells) is not int or min_cells < 2
            or type(max_cells) is not int or max_cells < 2):
        raise ValueError('Invalid split fractions or response-cell limits')
    fingerprint = _source_fingerprint(reader)
    by_target = defaultdict(list)
    for row, cell in enumerate(reader.cells):
        if cell['is_control'] == '0' and cell['pool'] == 'response_unassigned' and int(cell['library_total']) > 0:
            by_target[cell['target']].append(row)
    eligible = sorted(target for target, rows in by_target.items() if len(rows) >= min_cells)
    if not eligible:
        raise ValueError('No VCC 2025 response targets pass min_response_cells')
    shuffled = stable_rng(seed, 'vcc2025:target_split').permutation(eligible).tolist()
    n_test = max(1, int(len(eligible) * test_fraction)) if test_fraction else 0
    n_development = max(1, int((len(eligible) - n_test) * development_fraction)) if development_fraction else 0
    if n_test + n_development >= len(eligible):
        raise ValueError('Too few eligible targets for the requested target holdouts')
    test = set(shuffled[:n_test])
    development = set(shuffled[n_test:n_test + n_development])
    train = set(shuffled[n_test + n_development:])
    groups = {partition: {} for partition in PARTITIONS}
    for target in sorted(eligible):
        ordered = sorted(by_target[target], key=lambda row: reader.cells[row]['cell_id'])
        rows = stable_rng(seed, f'vcc2025:{target}:cells').permutation(ordered).tolist()
        if target in test:
            groups['test'][target] = rows
        elif target in development:
            groups['development_unseen'][target] = rows
        else:
            n_seen = max(1, int(len(rows) * cell_fraction)) if cell_fraction else 0
            n_seen = min(n_seen, len(rows) - 1)
            groups['development_seen'][target] = rows[:n_seen]
            groups['train'][target] = rows[n_seen:n_seen + max_cells]
    manifest = {
        'schema_version': 1, 'seed': seed, 'source': fingerprint,
        'dataset': 'vcc2025', 'release_split': 'train', 'context': 'H1',
        'train_targets': sorted(train), 'development_targets': sorted(development), 'test_targets': sorted(test),
        'fractions': {'test_targets': test_fraction, 'development_targets_of_remaining': development_fraction,
                      'development_cells_per_seen_target': cell_fraction},
        'min_response_cells': min_cells, 'max_training_cells_per_target': max_cells,
        'excluded_targets': sorted(set(by_target) - set(eligible)), 'groups': groups,
        'control_policy': 'VCC 2025 fit controls for reference statistics only; model controls are VCC 2026',
        'evaluation_policy': 'Target-held-out responses never enter losses, reference fitting, or checkpoint selection',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest


def load_splits(reader, path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema_version') != 1 or manifest.get('source') != _source_fingerprint(reader):
        raise ValueError('Frozen split source fingerprint does not match the prepared VCC 2025 source')
    train, development, test = (set(manifest[name]) for name in ('train_targets', 'development_targets', 'test_targets'))
    if not train or train & development or train & test or development & test:
        raise ValueError('Split target sets must be nonempty for training and disjoint across target holdouts')
    expected = {'train': train, 'development_seen': train, 'development_unseen': development, 'test': test}
    seen = set()
    for partition in PARTITIONS:
        groups = manifest['groups'][partition]
        if set(groups) != expected[partition]:
            raise ValueError(f'Split target membership mismatch: {partition}')
        for target, rows in groups.items():
            if partition == 'train' and not rows:
                raise ValueError('Training targets require response rows')
            for row in rows:
                if type(row) is not int or not 0 <= row < len(reader.cells) or row in seen:
                    raise ValueError('Split contains invalid, duplicate, or overlapping response rows')
                cell = reader.cells[row]
                if (cell['target'] != target or cell['context'] != 'H1' or cell['is_control'] != '0'
                        or cell['pool'] != 'response_unassigned' or int(cell['library_total']) <= 0):
                    raise ValueError('Split contains a control, wrong target/context, or excluded response')
                seen.add(row)
    return manifest


def split_summary(manifest):
    return {partition: {'targets': len(groups), 'cells': sum(map(len, groups.values()))}
            for partition, groups in manifest['groups'].items()}
