"""Portable regression tests: python -m unittest discover -s tests -v."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.dataset import SequenceDataset
from data.prepper import DataPrepper
from service.bundle import BundleError, _check_config_shapes
from service.forecast_input import build_forecast_input

ROOT = Path(__file__).resolve().parents[1]


class ForecastInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.features = ['cbg', 'bolus', 'carbInput']
        self.frame = pd.DataFrame({
            '5minute_intervals_timestamp': pd.date_range(
                '2026-09-01', periods=96, freq='5min', tz='UTC'),
            'cbg': np.linspace(90, 180, 96),
            'bolus': np.where(np.arange(96) % 12 == 0, 2.0, 0.0),
            'carbInput': np.where(np.arange(96) % 24 == 0, 30.0, 0.0),
        })
        for name, columns in [('scaler_input', self.features), ('scaler_target', ['cbg'])]:
            scaler = StandardScaler().fit(self.frame[columns])
            joblib.dump(scaler, Path(self.directory.name) / (name + '.pkl'))
        self.hp = dict(feature_window=48, forecast_steps=8, patch_size=4,
                       history_of_days=0, context_limit=None, baseline=False,
                       n_features=3)
        self.run = dict(features=self.features, targets=self.features,
                        allowed_missing_values_rate=[1, 1, 1],
                        allowed_missing_values_rate_target=[1, 1, 1],
                        fill_types=[-9, -9, -9], disabled_covariates=[0, 0, 0],
                        rolling_mean_window=[1, 12, 12])

    def build(self, frame, stride=None):
        datasets = []

        def make_prepper(**kwargs):
            if stride is not None:
                kwargs['step'] = stride
            return DataPrepper(**kwargs)

        def capture_dataset(**kwargs):
            dataset = SequenceDataset(**kwargs)
            datasets.append(dataset)
            return dataset

        with patch('service.forecast_input.DataPrepper', side_effect=make_prepper), \
                patch('data.prepper.SequenceDataset', side_effect=capture_dataset), \
                contextlib.redirect_stderr(io.StringIO()):
            prepper, window, origin = build_forecast_input(
                frame, {'age': 42}, 'patient', self.hp, self.run, self.directory.name)
        return prepper, window, origin, datasets[0]

    def test_latest_observation_and_training_length(self):
        original = self.frame.copy(deep=True)
        prepper, window, origin, dataset = self.build(self.frame)
        self.assertEqual(window.shape, (40, 3))
        self.assertEqual(dataset[0][0].shape[0], len(window))
        self.assertEqual(origin, self.frame.iloc[-1, 0])
        expected = prepper.scaler_x.scaler.transform(self.frame[self.features])[-40:, 0]
        np.testing.assert_array_equal(window[:, 0], expected)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_short_history_is_left_padded(self):
        frame = self.frame.tail(24).copy()
        prepper, window, origin, _ = self.build(frame)
        np.testing.assert_array_equal(window[:16], np.full((16, 3), -9))
        expected = prepper.scaler_x.scaler.transform(frame[self.features])[:, 0]
        np.testing.assert_array_equal(window[-24:, 0], expected)
        self.assertEqual(origin, frame.iloc[-1, 0])

    def test_stride_preserves_preprocessing_and_limits_scan(self):
        for count in (24, 40, 48, 96):
            with self.subTest(rows=count):
                frame = self.frame.tail(count).copy()
                _, fast, origin, fast_dataset = self.build(frame)
                _, reference, reference_origin, slow_dataset = self.build(frame, stride=1)
                np.testing.assert_array_equal(fast, reference)
                np.testing.assert_array_equal(fast_dataset.input_data, slow_dataset.input_data)
                self.assertEqual(origin, reference_origin)
                candidates = range(0, len(fast_dataset.input_data)
                                   - fast_dataset.sequence_length - fast_dataset.forecast_steps,
                                   fast_dataset.step)
                self.assertLessEqual(len(candidates), 1)

    def test_invalid_history_length_rejected(self):
        for window in (8, 4, 49):
            with self.subTest(feature_window=window), self.assertRaises(BundleError):
                _check_config_shapes(dict(self.hp, feature_window=window), self.features)
        _check_config_shapes(self.hp, self.features)


class PipelineCompatibilityTests(unittest.TestCase):
    def test_forecast_import_without_evaluation_dependencies(self):
        code = '''
import importlib.abc
import sys
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('cg_ega', 'evaluation', 'pipeline.evaluation'):
            raise AssertionError('Forecast imported ' + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from service.forecaster import Forecaster
import pipeline
assert callable(pipeline.main)
assert 'pipeline.evaluation' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                       capture_output=True, text=True)

    def test_existing_exports_and_legacy_dispatch(self):
        import pipeline
        from pipeline import orchestrator, training

        for name in ('train_model', 'create_temporal_split', 'train_model_glucobench'):
            self.assertIs(getattr(pipeline, name), getattr(training, name))
        evaluation = types.ModuleType('pipeline.evaluation')
        names = ('evaluate_model', 'evaluate_model_glucobench', 'plot_per_horizon')
        for name in names:
            setattr(evaluation, name, lambda: None)
        # Exercise the lazy package exports without requiring optional CG-EGA.
        try:
            with patch.dict(sys.modules, {'pipeline.evaluation': evaluation}):
                for name in names:
                    self.assertIs(getattr(pipeline, name), getattr(evaluation, name))
        finally:
            for name in names:
                pipeline.__dict__.pop(name, None)
        with self.assertRaises(AttributeError):
            getattr(pipeline, 'not_a_pipeline_export')
        with patch.object(orchestrator, 'init_experiment_directory',
                          side_effect=RuntimeError('legacy dispatch')) as initialize:
            with self.assertRaisesRegex(RuntimeError, 'legacy dispatch'):
                pipeline.main({'run_config': {}})
            initialize.assert_called_once()


if __name__ == '__main__':
    unittest.main()
