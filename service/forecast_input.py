"""Build the model input for one forecast request.

Preprocessing is delegated to ``DataPrepper`` **unmodified**, so smoothing,
zero-to-missing conversion, scaling, sentinel fills and front-padding are exactly
what training and evaluation use.  The only thing done here is choosing which
window to feed: the final ``feature_window - forecast_steps`` rows, ending at the
forecast origin, matching the history length supplied during training.

Nothing is truncated from the end of the sequence, so the model receives the
latest observation and predicts genuinely future steps.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from data.prepper import DataPrepper
from service.bundle import BundleError

TIMESTAMP_COLUMN = '5minute_intervals_timestamp'


class ForecastOnlyScaler(StandardScaler):
    """A scaler that refuses to be fitted.

    ``Scaler`` fits and saves a new scaler whenever its ``.pkl`` is absent
    ([data/scaler.py] ``fit_and_save``), which would silently scale a forecast by
    statistics derived from the request's own handful of rows *and* overwrite the
    bundle artifact.  Passing this class in as ``scaler_class_x`` / ``scaler_class_y``
    makes that impossible: ``fit_and_save`` calls ``fit``/``partial_fit`` **before**
    ``joblib.dump``, so the attempt raises before anything is written.

    On the normal path this instance is simply discarded — ``Scaler.load_scaler``
    rebinds ``self.scaler`` to the scaler loaded from disk, so transforms are done by
    an ordinary fitted ``StandardScaler``.

    This requires no change to shared code and no monkeypatching.
    """

    def fit(self, *args, **kwargs):
        raise BundleError(
            'Forecasting requires an existing fitted scaler; refusing to fit one on '
            'request data. The bundle artifact is missing or was removed mid-request.')

    def partial_fit(self, *args, **kwargs):
        raise BundleError(
            'Forecasting cannot fit a scaler; refusing to fit on request data. '
            'The bundle artifact is missing or was removed mid-request.')


def _assert_scalers_not_refitted(prepper, expected_scaler_params):
    """Consistency check that the scalers in use are the ones that were verified.

    This does **not** prove no fitting occurred — two different datasets can produce
    identical means and scales. Fitting is *prevented* by :class:`ForecastOnlyScaler`;
    this check catches a different failure, namely a bundle whose artifacts were
    swapped for other valid scalers between verification and use.
    """
    in_use = {'scaler_input': prepper.scaler_x.scaler, 'scaler_target': prepper.scaler_y.scaler}
    for name, scaler in in_use.items():
        expected = expected_scaler_params[name]
        actual_mean = np.asarray(getattr(scaler, 'mean_', []), dtype=float)
        actual_scale = np.asarray(getattr(scaler, 'scale_', []), dtype=float)
        if (actual_mean.shape != expected['mean'].shape
                or not np.array_equal(actual_mean, expected['mean'])
                or not np.array_equal(actual_scale, expected['scale'])):
            raise BundleError(
                f'{name} in use does not match the verified bundle (expected mean '
                f'{expected["mean"]}, got {actual_mean}). The artifact was replaced between '
                'verification and use; restart against an immutable bundle.')


def build_forecast_input(dataframe, metadata, user_id, hp_config, run_config, model_dir,
                         expected_scaler_params=None):
    """Return ``(prepper, window, origin)`` for one user.

    Args:
        dataframe: The user's dataframe from ``DataloaderMelissa``.
        metadata: That user's metadata dict (must be non-empty; see below).
        user_id: Participant key.
        hp_config: Bundle ``hp_config``.
        run_config: Bundle ``run_config``.
        model_dir: Directory holding the fitted scalers.

    Returns:
        tuple: ``(prepper, window, origin)`` where ``window`` is the scaled
        ``(feature_window - forecast_steps, n_features)`` array ending at ``origin``. The prepper
        is returned because its scaler performs the inverse transform.

    Note:
        ``model_dir`` is where ``Scaler`` looks for ``scaler_input.pkl`` /
        ``scaler_target.pkl``.  It loads them when present but **fits and saves new
        ones when absent** — so this function is only load-only if the bundle has
        been verified (see :mod:`service.bundle`).  Each call constructs a new
        ``DataPrepper``, which re-reads both scalers from disk; the warm runtime
        holds the model only.
    """
    if not metadata:
        # SequenceDataset returns a 3-tuple instead of a 4-tuple when metadata is
        # falsy, and the model was trained with the metadata block present.
        raise ValueError('Metadata must not be empty; unknown values are represented by '
                         'omitting individual fields, not by omitting the dict.')

    feature_window = hp_config['feature_window']

    prepper = DataPrepper(
        participants=[user_id],
        # DataPrepper mutates the frames it is given (zero -> NaN); pass a copy.
        dataframes={user_id: dataframe.copy()},
        feature_list=run_config['features'],
        target_list=run_config['targets'],
        allowed_missing_values_rate=run_config['allowed_missing_values_rate'],
        allowed_missing_values_rate_target=run_config['allowed_missing_values_rate_target'],
        forecast_steps=hp_config['forecast_steps'],
        # Separate instances, both refusing to fit: a missing artifact raises inside
        # Scaler.fit_and_save before joblib.dump, so the bundle cannot be overwritten.
        scaler_class_x=ForecastOnlyScaler(),
        scaler_class_y=ForecastOnlyScaler(),
        patch_size=hp_config['patch_size'],
        fill_types=run_config['fill_types'],
        experiment_path=model_dir,
        # Only input_data is used below. Check at most one candidate instead of
        # scanning every historical window for a dataset we never iterate over.
        step=max(1, len(dataframe)),
        sequence_length=feature_window,
        history_of_days=hp_config['history_of_days'],
        # No augmentation at inference time.
        mask_prob=0.0,
        chance_of_smbg=0.0,
        chance_feature_missing=0.0,
        metadata={user_id: metadata},
        mask_future_target_covariates=False,
        disabled_covariates=run_config['disabled_covariates'],
        context_limit=hp_config['context_limit'],
        baseline=hp_config['baseline'],
        rolling_mean_window=run_config['rolling_mean_window'],
    )

    # Checked immediately after construction, which is where Scaler opens the files.
    if expected_scaler_params is not None:
        _assert_scalers_not_refitted(prepper, expected_scaler_params)

    # input_data is front-padded by sequence_length, so it always has at least
    # feature_window rows even for a very short upload.
    prepared = prepper.make_features_and_targetpair().datasets[0].input_data
    history_length = feature_window - hp_config['forecast_steps']
    window = prepared[-history_length:]

    origin = dataframe[TIMESTAMP_COLUMN].iloc[-1]
    return prepper, window, origin
