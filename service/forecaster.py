"""Warm forecast service.

Construct one :class:`Forecaster` at process start and reuse it: building the
model and paying CUDA initialisation costs per request would dominate the
response time.

The runtime holds the **model** only. Scalers are re-read from disk by each
request's ``DataPrepper``, which is unmodified — see
:func:`service.forecast_input.build_forecast_input`.
"""

import logging
import os

from pipeline.config import build_forecast_config
from service.bundle import load_scaler_params, verify_model_bundle
from service.forecast_run import ForecastError, run_forecast

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(_ROOT, 'inference', 'test')
# Every forecast is kept by default; pass output_dir=None to keep results in memory.
DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, 'forecast_runs')


class Forecaster:
    """Holds a verified bundle and serves forecasts from it.

    Args:
        model_dir: Directory holding ``best_model.pth``, both scalers and
            ``model_config.json``. Verified at construction and treated as
            immutable thereafter.
        device: ``'cpu'`` (default) or ``'cuda'``. CPU serves a request in roughly
            150 ms and avoids a multi-second CUDA warm-up.
        output_dir: Writable root for per-run artifacts. Defaults to
            ``forecast_runs/`` at the repository root, so every forecast is kept.
            Pass ``None`` to keep results in memory only.
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device='cpu',
                 output_dir=DEFAULT_OUTPUT_DIR):
        self.model_dir = os.path.abspath(model_dir)
        self.device = device
        self.output_dir = os.path.abspath(output_dir) if output_dir else None

        config, model, digests = verify_model_bundle(self.model_dir, device=device)
        self.base_config = config
        self.runtime = {'model': model, 'device': device, 'digests': digests,
                        'model_dir': self.model_dir,
                        # Compared against the scalers each request actually loads,
                        # which is what proves no refit happened.
                        'scaler_params': load_scaler_params(self.model_dir)}
        logger.info('Forecaster ready: model_dir=%s device=%s', self.model_dir, device)

    def forecast(self, user_id, csv_path, horizon_minutes=30):
        """Forecast for one user from one CSV.

        Args:
            user_id: Must equal the CSV's ``user_id`` column.
            csv_path: Path to the uploaded CSV.
            horizon_minutes: 30, 60 or 120.

        Returns:
            service.result.ForecastResult

        Raises:
            dataloaders.csv_validation.InvalidCSV: Malformed or unusable input.
            service.bundle.BundleError: The bundle is missing or has changed.
            ForecastError: Unsupported horizon or invalid user id.
        """
        config = build_forecast_config(
            self.base_config, model_dir=self.model_dir, csv_path=csv_path,
            user_id=user_id, horizon_minutes=horizon_minutes, output_dir=self.output_dir)

        # Goes through the pipeline entry point, which dispatches to run_forecast.
        from pipeline.orchestrator import main

        return main(config, runtime=self.runtime)


__all__ = ['Forecaster', 'ForecastError', 'run_forecast']
