"""Model-bundle verification.

Why this module exists: ``data.scaler.Scaler`` **fits and saves a new scaler**
whenever the expected ``.pkl`` is absent (``data/scaler.py``), and the forecast
path uses it unmodified.  Pointing ``experiment_path`` at the model directory is
therefore *not* a load-only guarantee on its own — if an artifact goes missing a
``StandardScaler`` would be silently fitted on a single request's rows and the
forecast would be confidently wrong instead of failing.

The guarantee is supplied here instead: the bundle is verified at startup and
re-checked against those digests before each forecast, so it cannot change under
a running process.
"""

import hashlib
import json
import os

import joblib
import numpy as np
import torch

ARTIFACTS = ('best_model.pth', 'scaler_input.pkl', 'scaler_target.pkl', 'model_config.json')


class BundleError(RuntimeError):
    """Raised when a model bundle is missing, unusable, or has changed."""


def _sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_files_readable(model_dir):
    """Presence, non-emptiness and non-null-content for every artifact."""
    digests = {}
    for name in ARTIFACTS:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise BundleError(
                f'Model artifact missing: {path}. Refusing to continue: without it the '
                'unmodified Scaler would fit a new scaler on request data and return a '
                'confidently wrong forecast.')
        if os.path.getsize(path) == 0:
            raise BundleError(f'Model artifact is empty: {path}.')
        with open(path, 'rb') as handle:
            if not any(handle.read(1 << 20)):
                raise BundleError(
                    f'Model artifact contains only null bytes: {path}. The file has the correct '
                    'size but no content. Re-transfer it and verify the checksum at the source.')
        digests[name] = _sha256(path)
    return digests


def _check_scalers(model_dir, features, test_target):
    """Scaler/config compatibility, including exact feature ORDER.

    ``StandardScaler`` is positional: a reordered feature list would silently apply
    the wrong mean and scale to each channel.  The scalers here were fitted on a
    named DataFrame, so ``feature_names_in_`` lets us check order exactly.
    """
    try:
        scaler_input = joblib.load(os.path.join(model_dir, 'scaler_input.pkl'))
        scaler_target = joblib.load(os.path.join(model_dir, 'scaler_target.pkl'))
    except Exception as exc:
        raise BundleError(f'Scaler artifacts could not be loaded from {model_dir}: {exc}') from exc

    if scaler_input.n_features_in_ != len(features):
        raise BundleError(
            f'scaler_input was fitted on {scaler_input.n_features_in_} features but the config '
            f'lists {len(features)}: {features}.')
    if scaler_target.n_features_in_ != 1:
        raise BundleError(
            f'scaler_target must be fitted on exactly one feature; got {scaler_target.n_features_in_}.')

    names = getattr(scaler_input, 'feature_names_in_', None)
    if names is not None and list(names) != list(features):
        raise BundleError(
            f'Feature ORDER mismatch: scaler_input was fitted on {list(names)} but the config '
            f'lists {list(features)}. StandardScaler is positional, so this would apply the '
            'wrong mean/scale to each channel.')

    target_names = getattr(scaler_target, 'feature_names_in_', None)
    if target_names is not None and list(target_names) != [test_target]:
        raise BundleError(
            f'scaler_target was fitted on {list(target_names)} but test_target is {test_target!r}.')

    return scaler_input, scaler_target


def _check_config_shapes(hp_config, features):
    if hp_config['n_features'] != len(features):
        raise BundleError(
            f"hp_config.n_features={hp_config['n_features']} disagrees with "
            f'{len(features)} configured features.')
    if hp_config['feature_window'] % hp_config['patch_size'] != 0:
        raise BundleError(
            f"feature_window={hp_config['feature_window']} is not a multiple of "
            f"patch_size={hp_config['patch_size']}; PeriodicityReshape would raise.")


def load_config(model_dir):
    """Read and return the bundle's ``model_config.json``."""
    with open(os.path.join(model_dir, 'model_config.json')) as handle:
        return json.load(handle)


def load_scaler_params(model_dir):
    """Return the fitted parameters of the verified scalers.

    These are compared against whatever ``DataPrepper`` actually loads, which is the
    only way to prove no refit happened: verifying files beforehand cannot, because
    the scalers are opened later in the request.
    """
    params = {}
    for name in ('scaler_input', 'scaler_target'):
        scaler = joblib.load(os.path.join(model_dir, f'{name}.pkl'))
        params[name] = {'mean': np.asarray(scaler.mean_, dtype=float).copy(),
                        'scale': np.asarray(scaler.scale_, dtype=float).copy()}
    return params


def bundle_version(model_dir, digests):
    """Identify a bundle by its content, not by its directory name.

    A directory basename such as ``test`` is unchanged when the bundle inside it is
    replaced, so it cannot identify which weights produced a forecast.
    """
    combined = hashlib.sha256()
    for name in ARTIFACTS:
        combined.update(digests[name].encode())
    return f'{os.path.basename(os.path.normpath(model_dir))}@{combined.hexdigest()[:12]}'


def build_model(hp_config, model_dir, device='cpu'):
    """Construct the architecture and load the checkpoint with ``strict=True``."""
    # Imported lazily: constructing Args/Model pulls in the trainer stack.
    from architectures.iTransformerMasked import Model
    from trainers.exp_long_term_forecasting import Args

    model = Model(Args(hp_config))
    state = torch.load(os.path.join(model_dir, 'best_model.pth'),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    expected = hp_config['forecast_steps']
    produced = model.forecast_projector.projector.out_features
    if produced != expected:
        raise BundleError(
            f'forecast head produces {produced} steps but hp_config.forecast_steps={expected}.')
    return model


def verify_model_bundle(model_dir, device='cpu'):
    """Verify a bundle end to end and return ``(config, model, digests)``.

    Raises:
        BundleError: If any artifact is missing, empty, all-null, unloadable, or
            inconsistent with the config it ships with.
    """
    model_dir = os.path.abspath(model_dir)
    digests = _check_files_readable(model_dir)

    config = load_config(model_dir)
    hp_config, run_config = config['hp_config'], config['run_config']
    features = run_config['features']

    _check_scalers(model_dir, features, run_config['test_target'])
    _check_config_shapes(hp_config, features)
    model = build_model(hp_config, model_dir, device=device)
    return config, model, digests


def assert_bundle_intact(model_dir, expected_digests):
    """Re-check a verified bundle; the artifacts must not have changed.

    Raises:
        BundleError: If an artifact is missing, unusable, or differs from startup.
    """
    digests = _check_files_readable(os.path.abspath(model_dir))
    for name, digest in expected_digests.items():
        if digests.get(name) != digest:
            raise BundleError(
                f'Model artifact changed while the service was running: {name}. The bundle must '
                'be immutable; restart against a verified bundle.')
    return digests
