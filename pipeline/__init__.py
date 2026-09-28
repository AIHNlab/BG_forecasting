"""Top-level ML pipeline: experiment setup, training, and evaluation.

Entry point is ``main(config)`` in ``pipeline.orchestrator``, which
loads data, trains a model, and runs evaluation based on the
``model_config.json`` experiment configuration.

Sub-modules:

- **orchestrator** — ``main()`` and ``init_experiment_directory()``.
- **training** — ``train_model()`` and ``train_model_glucobench()``.
- **evaluation** — ``evaluate_model()`` and ``evaluate_model_glucobench()``.
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import main, init_experiment_directory
from pipeline.training import train_model, create_temporal_split, train_model_glucobench

__all__ = [
    'main', 'init_experiment_directory', 'train_model', 'create_temporal_split',
    'train_model_glucobench', 'evaluate_model', 'evaluate_model_glucobench',
    'plot_per_horizon',
]


def __getattr__(name):
    # Forecasting does not need evaluation-only dependencies such as cg_ega.
    # Keep the existing package exports available when evaluation is requested.
    if name in ('evaluate_model', 'evaluate_model_glucobench', 'plot_per_horizon'):
        from importlib import import_module

        value = getattr(import_module('pipeline.evaluation'), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
