"""Forecast result container and serialization.

The response deliberately separates two quantities that are easy to conflate:

* ``forecast`` — a per-step trajectory over the **requested** horizon.
* ``event_risk`` — a single window-level score over a **fixed 60-minute** horizon,
  produced by the alarm heads. It is not a per-step probability, is not rescalable
  to the requested horizon, and is not a calibrated probability, so it is named
  ``event_risk`` rather than ``*_prob`` and carries its own ``horizon_minutes``.

The per-step CSV carries no event-risk column: repeating a window-level score on
every row is exactly the misreading this split exists to prevent.
"""

import csv
import io
from dataclasses import dataclass, field

import pandas as pd

SAMPLING_INTERVAL_MIN = 5
INTERVAL_Z = 1.96  # nominal 95%, uncalibrated
# The alarm heads were trained on the first 12 forecast steps; this is not adjustable.
EVENT_RISK_HORIZON_MIN = 60

CSV_COLUMNS = (
    # run_id leads so the per-step file joins to event_risk.csv on one key.
    'run_id', 'user_id', 'forecast_origin', 'prediction_time', 'minutes_ahead',
    'predicted_glucose_mgdl', 'pred_std_mgdl',
    'ci_low_mgdl', 'ci_high_mgdl', 'glycemic_zone',
)

# Event risk is one value per request, not per step, so it gets its own single-row
# file rather than a column repeated on every step of the trajectory.
EVENT_RISK_CSV_COLUMNS = (
    'run_id', 'user_id', 'forecast_origin', 'model_version',
    'event_horizon_minutes', 'scale',
    'hypo_threshold_mgdl', 'hyper_threshold_mgdl',
    'event_risk_hypo', 'event_risk_hyper',
)


def glycemic_zone(value, hypo_threshold, hyper_threshold):
    """Classify a predicted glucose value against the configured thresholds."""
    if value < hypo_threshold:
        return 'low'
    if value > hyper_threshold:
        return 'high'
    return 'in_range'


@dataclass
class ForecastResult:
    """One forecast, ready to serialize as JSON or CSV."""

    run_id: str
    user_id: str
    forecast_origin: pd.Timestamp
    model_version: str
    horizon_minutes: int
    steps: list
    event_risk_hypo: float
    event_risk_hyper: float
    hypo_threshold: float
    hyper_threshold: float
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            'run_id': self.run_id,
            'user_id': self.user_id,
            'forecast_origin': self.forecast_origin.isoformat(),
            'model_version': self.model_version,
            'forecast': {
                'horizon_minutes': self.horizon_minutes,
                'unit': 'mg/dL',
                'sampling_interval_minutes': SAMPLING_INTERVAL_MIN,
                'interval': {
                    'coverage': 'nominal_95',
                    # The probabilistic mean is not reported; it is not the point
                    # forecast, so the interval is slightly asymmetric about it.
                    'centred_on': 'probabilistic_mean',
                    'centred_on_predicted_glucose': False,
                    'calibrated': False,
                },
                'steps': self.steps,
            },
            'event_risk': {
                'horizon_minutes': EVENT_RISK_HORIZON_MIN,
                'thresholds_mgdl': {'hypo': self.hypo_threshold, 'hyper': self.hyper_threshold},
                'hypo': self.event_risk_hypo,
                'hyper': self.event_risk_hyper,
            },
            'warnings': self.warnings,
        }

    def to_csv(self):
        """Per-step CSV. Deliberately contains no event-risk column."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator='\n')
        writer.writeheader()
        origin = self.forecast_origin.isoformat()
        for step in self.steps:
            writer.writerow({
                'run_id': self.run_id,
                'user_id': self.user_id,
                'forecast_origin': origin,
                'prediction_time': step['prediction_time'],
                'minutes_ahead': step['minutes_ahead'],
                'predicted_glucose_mgdl': f"{step['predicted_glucose_mgdl']:.2f}",
                'pred_std_mgdl': f"{step['pred_std_mgdl']:.2f}",
                'ci_low_mgdl': f"{step['ci_low_mgdl']:.2f}",
                'ci_high_mgdl': f"{step['ci_high_mgdl']:.2f}",
                'glycemic_zone': step['glycemic_zone'],
            })
        return buffer.getvalue()

    def event_risk_to_csv(self):
        """Single-row CSV for the window-level event risk.

        Separate from the per-step file because this is one value for the whole
        request over a **fixed 60-minute** window — it neither varies per step nor
        follows the requested horizon. Rows from many runs stack into one table and
        join back to the per-step data on ``run_id``.
        """
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EVENT_RISK_CSV_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerow({
            'run_id': self.run_id,
            'user_id': self.user_id,
            'forecast_origin': self.forecast_origin.isoformat(),
            'model_version': self.model_version,
            'event_horizon_minutes': EVENT_RISK_HORIZON_MIN,
            'scale': 'uncalibrated_score',
            'hypo_threshold_mgdl': self.hypo_threshold,
            'hyper_threshold_mgdl': self.hyper_threshold,
            'event_risk_hypo': f'{self.event_risk_hypo:.6f}',
            'event_risk_hyper': f'{self.event_risk_hyper:.6f}',
        })
        return buffer.getvalue()


def build_steps(outputs, origin, n_steps, hypo_threshold, hyper_threshold):
    """Assemble the per-step trajectory for the requested horizon."""
    steps = []
    for index in range(n_steps):
        minutes_ahead = (index + 1) * SAMPLING_INTERVAL_MIN
        mean = float(outputs['mean'][index])
        std = float(outputs['std'][index])
        point = float(outputs['point'][index])
        steps.append({
            'prediction_time': (origin + pd.Timedelta(minutes=minutes_ahead)).isoformat(),
            'minutes_ahead': minutes_ahead,
            'predicted_glucose_mgdl': round(point, 2),
            'pred_std_mgdl': round(std, 2),
            # Built around the probabilistic mean, which is NOT reported: the variance
            # head was trained jointly with that mean, never with the deterministic
            # `forecast` head. The interval is therefore slightly asymmetric about
            # `predicted_glucose_mgdl`, which is correct and intended.
            'ci_low_mgdl': round(mean - INTERVAL_Z * std, 2),
            'ci_high_mgdl': round(mean + INTERVAL_Z * std, 2),
            'glycemic_zone': glycemic_zone(point, hypo_threshold, hyper_threshold),
        })
    return steps
