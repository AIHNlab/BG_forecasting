"""Dataloader for single-user MELISSA CGM CSV files.

Unlike the cohort dataloaders, this one describes exactly one participant and
performs no train/test split: everything lands in the test collections, which is
what the forecast path consumes.

``directory_path`` may be either the CSV file itself or a directory containing
exactly one CSV.

Metadata column names in these files already match the model's own token names
(``diagnosis_type``, ``biological_sex``, ``insulin_treatment``, ``age``, ``bmi``),
so they are passed through unchanged.  Note that several cohort dataloaders in
this package use ``diagonosis_type``/``device_type`` instead, which the model
does not read; that behaviour is deliberately left alone here.
"""

import os
from glob import glob

import numpy as np
import pandas as pd

from dataloaders.dataloader import Dataloader
from dataloaders.csv_validation import InvalidCSV, validate
from utils import calculate_total_cob, calculate_total_iob

ID_COLUMN = 'user_id'
TIMESTAMP_COLUMN = '5minute_intervals_timestamp'
# Columns the pipeline expects to exist, whether or not the upload supplied them.
EXPECTED_TIMESERIES = ('cbg', 'bolus', 'carbInput', 'basal', 'hr')


class DataloaderMelissa(Dataloader):
    """Loads one user's CGM CSV into the standard dataframe/metadata structures."""

    def __init__(self, directory_path, expected_user_id=None):
        super().__init__(directory_path)
        self.id_column = ID_COLUMN
        self.expected_user_id = expected_user_id
        self._metadata = {}

    def _resolve_csv_path(self):
        path = self.directory_path
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            matches = sorted(glob(os.path.join(path, '*.csv')))
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise InvalidCSV(f'No CSV file found in {path!r}.')
            raise InvalidCSV(f'Expected exactly one CSV in {path!r}; found {len(matches)}.')
        if os.path.isfile(path + '.csv'):
            return path + '.csv'
        raise InvalidCSV(f'No such CSV file or directory: {path!r}.')

    def _read_csv(self, path):
        """Read the CSV, translating parser failures into ``InvalidCSV``.

        ``user_id`` is read as a string: left to infer, pandas turns a valid id like
        ``"0001"`` into the integer ``1``, which then fails the request-id comparison.
        """
        try:
            return pd.read_csv(path, dtype={'user_id': str})
        except pd.errors.EmptyDataError as exc:
            raise InvalidCSV('CSV file is empty.') from exc
        except pd.errors.ParserError as exc:
            raise InvalidCSV(f'CSV could not be parsed: {exc}') from exc
        except UnicodeDecodeError as exc:
            raise InvalidCSV('CSV is not valid UTF-8 text.') from exc
        except ValueError as exc:
            raise InvalidCSV(f'CSV could not be read: {exc}') from exc
        except OSError as exc:
            raise InvalidCSV(f'CSV could not be opened: {exc}') from exc

    def load_data(self):
        """Read, validate and normalise the CSV into ``test``/``all`` collections."""
        df = self._read_csv(self._resolve_csv_path())
        df, user_id, metadata = validate(df, expected_user_id=self.expected_user_id)

        df = df.rename(columns={'timestamp': TIMESTAMP_COLUMN})
        for column in EXPECTED_TIMESERIES:
            if column not in df.columns:
                df[column] = np.nan
            else:
                df[column] = pd.to_numeric(df[column], errors='coerce')

        # Same derivation and parameters as the other dataloaders in this package.
        df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
        df['cob'] = calculate_total_cob(df['carbInput'].values, carb_absorption=0.8,
                                        ts_min=5, t_action_max_min=240)

        # One user, no cohort split: the forecast path reads the test collections.
        self.test_dataframes[user_id] = df
        self.all_dataframes[user_id] = df
        self._metadata = {user_id: metadata}

    def _get_dataset_specific_metadata(self):
        # Populated by load_data; the base class then adds the summary statistics.
        self.test_metadata.update({user: dict(meta) for user, meta in self._metadata.items()})

    def _get_dataframe(self, file, calculate_iob=True):
        pass
