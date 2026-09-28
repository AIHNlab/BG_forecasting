"""Run the model for one forecast request.

The model is called directly rather than through ``Exp_Long_Term_Forecast.test``,
which drops its final batch (``preds[:-1]``) — for a forecast the final window is
the only one, so nothing would be returned.  ``test()`` is left untouched.

Two distinct forecast heads are produced and kept separate:

* ``forecast`` — the deterministic head, trained with plain MSE.
* ``mean`` + ``variance`` — the probabilistic head, trained jointly with
  ``GaussianNLLLoss``.

The variance was fitted against ``mean``, never against ``forecast``, so intervals
belong around ``mean``.  Training clamps variance before the loss; inference in
``test()`` does not, so the clamp is reapplied here.
"""

import numpy as np
import torch

# The five metadata slots the model actually reads.
CATEGORICAL_SLOTS = ('diagnosis_type', 'biological_sex', 'insulin_treatment')
NUMERICAL_SLOTS = ('age', 'bmi')
VARIANCE_FLOOR = 1e-6


def build_model_metadata(metadata, device='cpu'):
    """Convert a metadata dict into the form the encoder expects.

    Absent slots are simply omitted; the model substitutes its own ``unknown_*``
    tokens, keeping the token count constant either way.
    """
    model_metadata = {}
    for slot in CATEGORICAL_SLOTS:
        if slot in metadata and metadata[slot] is not None:
            model_metadata[slot] = [str(metadata[slot])]
    for slot in NUMERICAL_SLOTS:
        if slot in metadata and metadata[slot] is not None:
            model_metadata[slot] = torch.tensor([float(metadata[slot])], device=device)
    return model_metadata


def run_forecast_model(model, window, metadata, prepper, hp_config, run_config, device='cpu'):
    """Run one forward pass and return everything in mg/dL.

    Args:
        model: The warm, evaluation-mode model.
        window: Scaled ``(feature_window, n_features)`` array ending at the origin.
        metadata: The user's metadata dict.
        prepper: The ``DataPrepper`` whose scaler inverts the transform.
        hp_config, run_config: Bundle configuration.
        device: Torch device.

    Returns:
        dict with ``point``, ``mean``, ``std`` (each ``forecast_steps`` long, mg/dL)
        and the two window-level alarm scores.
    """
    from architectures.iTransformerMasked import PeriodicityReshape

    n_features = len(run_config['features'])
    target = run_config['test_target']

    sequence = torch.tensor(np.asarray(window), dtype=torch.float32, device=device).unsqueeze(0)
    x = PeriodicityReshape(hp_config['patch_size'])(sequence, n_features, 'apply')

    with torch.inference_mode():
        _, forecast, mean, variance, hyperglycemia, hypoglycemia = model(
            x, None, None, None, metadata=build_model_metadata(metadata, device) or None)

    expected = hp_config['forecast_steps']
    if forecast.shape[-1] != expected:
        raise RuntimeError(f'Model returned {forecast.shape[-1]} forecast steps, expected {expected}.')

    # Training clamps variance before GaussianNLLLoss; test() omits this, which can
    # yield NaN from sqrt of a negative value.
    std_scaled = torch.sqrt(torch.clamp(variance[0], min=VARIANCE_FLOOR))

    inverse = prepper.scaler_x.inverse_transform_single_value
    point = np.array([inverse(float(v), target) for v in forecast[0]], dtype=float)
    mean_mgdl = np.array([inverse(float(v), target) for v in mean[0]], dtype=float)
    # sigma is a spread, not a location: scale it, never shift it.
    scale = float(prepper.scaler_x.scaler.scale_[run_config['features'].index(target)])
    std_mgdl = std_scaled.detach().cpu().numpy().astype(float) * scale

    if not (np.all(np.isfinite(point)) and np.all(np.isfinite(mean_mgdl))
            and np.all(np.isfinite(std_mgdl))):
        raise RuntimeError('Model produced a non-finite forecast.')

    return {
        'point': point,
        'mean': mean_mgdl,
        'std': std_mgdl,
        'event_risk_hypo': float(torch.sigmoid(hypoglycemia).item()),
        'event_risk_hyper': float(torch.sigmoid(hyperglycemia).item()),
    }
