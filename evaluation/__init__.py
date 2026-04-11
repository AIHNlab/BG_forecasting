"""Forecast and alarm evaluation metrics.

This package provides three layers of evaluation:

- **evaluation.forecast** — per-horizon RMSE/MAE (overall, hypo, hyper,
  normo), CG-EGA analysis, and uncertainty calibration (PICP / PICE).
- **evaluation.alarm** — event-level alarm precision/recall, false-alarm
  rate, calibration curves, and visualization of predicted vs. actual
  hypo/hyper events.
- **evaluation.metrics** — ``Evaluator`` class for quick model
  inference + aggregate RMSE on a test loader.
"""

from evaluation.alarm import (
    run_evaluation,
    compute_calibration_curve,
    plot_calibration_curve,
    aggregate_calibration_data,
    compute_event_level_metrics,
    filter_binary_events,
)
from evaluation.forecast import (
    plot_and_evaluate_horizon,
    evaluate_cg_ega_horizon,
    evaluate_uncertainty_calibration,
    create_calibration_analysis,
)
from evaluation.metrics import Evaluator
