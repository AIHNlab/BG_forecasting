"""Data loading, preprocessing, scaling, and PyTorch dataset construction.

This package contains the full data pipeline:

- **DataHandler** — loads raw datasets via pluggable dataloaders, caches to
  pickle/CSV under ``standardized_datasets/``.
- **DataPrepper** — selects features/targets, scales values, handles missing
  data, and produces ``SequenceDataset`` instances ready for training.
- **SequenceDataset** — PyTorch ``Dataset`` that yields windowed
  (sequence, reconstruction_target, forecast_target) tuples with patch
  masking, covariate dropout, and missing-value sentinel handling.
- **Scaler** — wraps a scikit-learn scaler, fitted once on training data
  and persisted to disk for reproducible transforms.
"""

from data.handler import DataHandler, CompactArrayEncoder
from data.prepper import DataPrepper
from data.dataset import SequenceDataset, CustomDataset, EventBalancedSampler
from data.scaler import Scaler
