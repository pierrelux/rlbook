"""Post-experiment checks on fixed, executed control and disturbance histories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from thermal_reactor import fine_rollout_diagnostics
from thermal_surrogate import Surrogate
from thermal_control import paired_interval


def audit(output_dir):
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / 'evaluation.json').read_text())
    model, _ = Surrogate.load(output_dir / 'surrogate.npz')
    details = []
    for row in summary['rows']:
        path = output_dir / f"trial-{row['trial']}-{row['controller']}.npz"
        with np.load(path) as data:
            x, u = data['fields'], data['controls']
            gust = data['disturbances'][:len(u)]
            if not len(u):
                continue
            pred = np.asarray(model.step(x[:-1], u, gust))
            error = pred - x[1:]
            trace = np.asarray(fine_rollout_diagnostics(x[0], u, gust, model.config))
            metrics = {
                'prediction_temperature_rmse_K': float(np.sqrt(np.mean(error[:, 0]**2))),
                'prediction_outlet_conversion_mae': float(np.abs(error[:, 1, :, -1].mean(axis=-1)).mean()),
                'prediction_peak_underprediction_K': float(np.maximum(x[1:, 0].max(axis=(-2, -1)) - pred[:, 0].max(axis=(-2, -1)), 0).max()),
                'fine_maximum_temperature_K': float(trace[:, 0].max()),
                'fine_minimum_outlet_conversion': float(trace[:, 1].min()),
                'fine_quality_violation_duration_s': float(np.sum(trace[:, 1] < model.config.conversion_target) * .1),
                'fine_temperature_violation_duration_s': float(np.sum(trace[:, 0] > model.config.temperature_limit) * .1),
                'fine_temperature_residual_K': float(trace[:, 0].max() - model.config.temperature_limit),
                'fine_conversion_residual': float(model.config.conversion_target - trace[:, 1].min()),
                'fine_sampling_interval_s': .1,
                'realized_disturbance_sha256': hashlib.sha256(data['disturbances'].tobytes()).hexdigest(),
            }
            row.update(metrics)
            details.append({'trial': row['trial'], 'controller': row['controller'], **metrics})
    for trial in sorted({r['trial'] for r in details}):
        digests = {r['realized_disturbance_sha256'] for r in details if r['trial'] == trial}
        if len(digests) != 1:
            raise AssertionError('Controllers did not use paired realized inlet histories')
    comparisons = []
    keys = ('normalized_heating_energy_s', 'minimum_outlet_conversion', 'maximum_temperature_K',
            'quality_violation_duration_s', 'temperature_violation_duration_s', 'planning_runtime_s',
            'prediction_temperature_rmse_K', 'fine_quality_violation_duration_s', 'fine_temperature_violation_duration_s')
    for left, right in (('stochastic', 'constant'), ('mean', 'constant'), ('stochastic', 'mean')):
        values = {key: [] for key in keys}
        for trial in sorted({r['trial'] for r in summary['rows']}):
            rows = {r['controller']: r for r in summary['rows'] if r['trial'] == trial}
            if not (rows[left]['completed'] and rows[right]['completed']):
                continue
            for key in keys:
                values[key].append(rows[left][key] - rows[right][key])
        comparisons.append({'difference': f'{left} minus {right}',
                            'complete_pair_count': len(values[keys[0]]),
                            'metrics': {k: paired_interval(v) if v else None for k, v in values.items()}})
    summary['paired_comparisons'] = comparisons
    if 'surrogate_checkpoint' in summary:
        summary['surrogate_checkpoint']['status'] = summary['surrogate_status']
    summary['diagnostic_note'] = ('Constraint checks during planning use 2 s output times. '
        'Post-experiment audits replay every executed control on a 0.1 s output grid. '
        'Neither grid provides a continuous-time or stochastic feasibility guarantee.')
    (output_dir / 'evaluation.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    result = {'runs': details, 'paired_disturbances_verified': True, 'paired_comparisons': comparisons,
              'diagnostic_note': summary['diagnostic_note']}
    (output_dir / 'execution_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'audited_runs': len(details), 'paired_disturbances_verified': True}), flush=True)


def audit_dataset(data_dir, output_dir):
    """Verify episode isolation and independently recompute training statistics."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    meta = json.loads((data_dir / 'dataset.json').read_text())
    folds = [set(meta[key]) for key in ('train_episodes', 'validation_episodes', 'test_episodes')]
    if (len(set.union(*folds)) != meta['episodes']
            or any(folds[i] & folds[j] for i, j in ((0, 1), (0, 2), (1, 2)))):
        raise AssertionError('Training, validation, and test episodes must form a disjoint partition')
    fields = np.load(data_dir / 'fields.npy', mmap_mode='r')
    total, squares, count = np.zeros(2), np.zeros(2), 0
    for i in meta['train_episodes']:
        block = np.asarray(fields[i], np.float64)
        total += block.sum(axis=(0, 2, 3)); squares += (block**2).sum(axis=(0, 2, 3))
        count += block.shape[0] * block.shape[2] * block.shape[3]
    mean = total / count
    std = np.maximum(np.sqrt(squares / count - mean**2), [1., .01])
    model, _ = Surrogate.load(output_dir / 'surrogate.npz')
    np.testing.assert_allclose(model.norm['mean'], mean, rtol=1e-6)
    np.testing.assert_allclose(model.norm['std'], std, rtol=1e-6)
    if model.config.to_dict() != meta['config']:
        raise AssertionError('Dataset and checkpoint model configurations differ')
    result = {'disjoint_whole_episode_splits': True, 'split_sizes': [len(f) for f in folds],
              'training_only_normalization_verified': True, 'channel_means': mean.tolist(),
              'channel_stds': std.tolist(), 'training_seed': meta['seed']}
    (output_dir / 'dataset_audit.json').write_text(json.dumps(result, indent=2))
    return result
