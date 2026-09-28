"""Forecast execution, reached from ``pipeline.orchestrator.main`` in forecast mode.

``main(config)`` dispatches here before ``init_experiment_directory`` runs, so a
forecast request never writes into the shared model directory.  Everything this
module writes goes under the per-request ``output_dir``.

A ``runtime`` may be supplied by :class:`service.forecaster.Forecaster` so the
model is not rebuilt per request; when it is ``None`` the model is loaded on the
spot, which keeps ``main(config)`` usable standalone.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

from dataloaders.dataloader import Dataloader
from dataloaders.dataloader_melissa import DataloaderMelissa
from service import bundle as bundle_module
from service.forecast_input import build_forecast_input
from service.forecast_model import run_forecast_model
from service.result import ForecastResult, build_steps

SAMPLING_INTERVAL_MIN = 5
SUPPORTED_HORIZONS = (30, 60, 120)
USER_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


class ForecastError(RuntimeError):
    """Raised when a forecast request cannot be served."""


def safe_run_directory(output_root, user_id, run_id):
    """Build ``<output_root>/<user_id>/<run_id>`` with the user id validated first.

    ``user_id`` is used as a single path segment, so it is checked against a
    pattern admitting no separators, dots or null bytes *before* any path is
    constructed, and the result is asserted to stay inside ``output_root``.
    """
    if not USER_ID_PATTERN.match(user_id):
        raise ForecastError(
            f'Invalid user_id {user_id!r}: expected 1-64 characters from [A-Za-z0-9_-].')

    output_root = os.path.abspath(output_root)
    directory = os.path.abspath(os.path.join(output_root, user_id, run_id))
    if os.path.commonpath([output_root, directory]) != output_root:
        raise ForecastError('Resolved output directory escapes the output root.')
    return directory


def _resolve_horizon(horizon_minutes, forecast_steps):
    if horizon_minutes not in SUPPORTED_HORIZONS:
        raise ForecastError(
            f'Unsupported horizon {horizon_minutes}; expected one of {list(SUPPORTED_HORIZONS)}.')
    n_steps = horizon_minutes // SAMPLING_INTERVAL_MIN
    if n_steps > forecast_steps:
        raise ForecastError(
            f'Horizon {horizon_minutes} min needs {n_steps} steps but the model produces '
            f'{forecast_steps}.')
    return n_steps


def run_forecast(config, runtime=None):
    """Run one forecast and return a :class:`~service.result.ForecastResult`."""
    hp_config, run_config = config['hp_config'], config['run_config']

    model_dir = os.path.abspath(run_config['model_dir'])
    csv_path = run_config['csv_path']
    user_id = str(run_config['user_id'])
    horizon_minutes = int(run_config.get('horizon_minutes', 30))
    output_root = run_config.get('output_dir')

    n_steps = _resolve_horizon(horizon_minutes, hp_config['forecast_steps'])
    # Validated before any computation, not only when a run directory is created.
    if not USER_ID_PATTERN.match(user_id):
        raise ForecastError(
            f'Invalid user_id {user_id!r}: expected 1-64 characters from [A-Za-z0-9_-].')

    # The unmodified Scaler fits a new scaler when an artifact is missing, so the
    # bundle is re-verified before it is used.
    if runtime is not None:
        digests = bundle_module.assert_bundle_intact(model_dir, runtime['digests'])
        model = runtime['model']
        device = runtime['device']
        scaler_params = runtime['scaler_params']
    else:
        _, model, digests = bundle_module.verify_model_bundle(model_dir)
        device = 'cpu'
        scaler_params = bundle_module.load_scaler_params(model_dir)

    loader = DataloaderMelissa(csv_path, expected_user_id=user_id)
    loader.load_data()
    Dataloader.get_metadata(loader)

    if user_id not in loader.test_dataframes:
        raise ForecastError(f'CSV did not yield data for user {user_id!r}.')
    dataframe = loader.test_dataframes[user_id]
    metadata = loader.test_metadata[user_id]

    prepper, window, origin = build_forecast_input(
        dataframe, metadata, user_id, hp_config, run_config, model_dir,
        expected_scaler_params=scaler_params)
    outputs = run_forecast_model(
        model, window, metadata, prepper, hp_config, run_config, device=device)

    hypo_threshold = hp_config.get('hypoglycemia_threshold', 70)
    hyper_threshold = hp_config.get('hyperglycemia_threshold', 180)
    steps = build_steps(outputs, origin, n_steps, hypo_threshold, hyper_threshold)
    if len(steps) != n_steps:
        raise ForecastError(f'Expected {n_steps} steps for a {horizon_minutes} min horizon, '
                            f'built {len(steps)}.')

    warnings = []
    observed = len(dataframe)
    history_length = hp_config['feature_window'] - hp_config['forecast_steps']
    if observed < history_length:
        warnings.append(
            f'History is {observed} samples; the model input was left-padded with '
            f'{history_length - observed} missing-value steps.')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:12]
    result = ForecastResult(
        run_id=run_id,
        user_id=user_id,
        forecast_origin=origin,
        model_version=bundle_module.bundle_version(model_dir, digests),
        horizon_minutes=horizon_minutes,
        steps=steps,
        event_risk_hypo=outputs['event_risk_hypo'],
        event_risk_hyper=outputs['event_risk_hyper'],
        hypo_threshold=hypo_threshold,
        hyper_threshold=hyper_threshold,
        warnings=warnings,
    )

    if output_root:
        directory = safe_run_directory(output_root, user_id, run_id)
        os.makedirs(directory, exist_ok=False)
        with open(os.path.join(directory, 'forecast.json'), 'x') as handle:
            json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
        with open(os.path.join(directory, 'forecast.csv'), 'x') as handle:
            handle.write(result.to_csv())
        # Own file: one row per run, not a value repeated on every trajectory step.
        with open(os.path.join(directory, 'event_risk.csv'), 'x') as handle:
            handle.write(result.event_risk_to_csv())

    return result
