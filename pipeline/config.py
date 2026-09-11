"""Per-request configuration for forecast runs.

Configs are built in memory from the bundle's ``model_config.json``; nothing is
written back, so a request can never rewrite a checked-in config or race another
request through the shared experiment directory.
"""

import copy
import os

FORECAST_MODE = 'forecast'


def build_forecast_config(base_config, model_dir, csv_path, user_id,
                          horizon_minutes=30, output_dir=None):
    """Return a config that drives one forecast through ``pipeline.orchestrator.main``.

    Args:
        base_config: The bundle's config; deep-copied, never mutated.
        model_dir: Read-only directory holding the checkpoint and scalers.
        csv_path: The user's uploaded CSV.
        user_id: Requesting user; must match the CSV's ``user_id`` column.
        horizon_minutes: 30, 60 or 120.
        output_dir: Writable root for per-run artifacts, or ``None`` to write nothing.

    Returns:
        dict: A config whose ``run_config.mode`` is ``'forecast'``.
    """
    config = copy.deepcopy(base_config)
    run_config = config['run_config']

    run_config['mode'] = FORECAST_MODE
    run_config['model_dir'] = os.path.abspath(model_dir)
    run_config['csv_path'] = os.fspath(csv_path)
    run_config['user_id'] = str(user_id)
    run_config['horizon_minutes'] = int(horizon_minutes)
    run_config['output_dir'] = os.path.abspath(output_dir) if output_dir else None

    # Forecasting neither trains nor evaluates; these keep the legacy flags coherent
    # for anything that inspects them.
    run_config['train'] = False
    run_config['test'] = False
    return config
