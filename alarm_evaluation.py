"""
Backwards-compatibility shim. Canonical location: evaluation/alarm.py
"""
from evaluation.alarm import (
    run_evaluation,
    compute_calibration_curve,
    plot_calibration_curve,
    aggregate_calibration_data,
    compute_event_level_metrics,
    filter_binary_events,
)
