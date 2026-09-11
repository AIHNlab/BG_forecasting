"""Validation for single-user CGM CSV uploads.

Used by :class:`~dataloaders.dataloader_melissa.DataloaderMelissa`.  Every check
raises :class:`InvalidCSV` with a message safe to return to an API client.

The accepted metadata values mirror the tokens the trained model actually holds
(``architectures.iTransformerMasked.EncoderModel.categorical_tokens``).  Values
outside these sets are rejected rather than silently degraded to ``unknown_*``,
which the model would otherwise do without any error.
"""

import re

import numpy as np
import pandas as pd

SAMPLING_INTERVAL_MIN = 5
MIN_ROWS = 24

# user_id is used as a single path segment downstream, so it is validated here —
# before any computation — rather than only when a run directory is created.
USER_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,64}$')

TIMESTAMP_COLUMNS = ('timestamp', '5minute_intervals_timestamp')
REQUIRED_COLUMNS = ('user_id', 'cbg')
TIMESERIES_COLUMNS = ('cbg', 'bolus', 'carbInput', 'basal')

CATEGORICAL_METADATA = {
    'diagnosis_type': {'type1', 'type2', 'prediabetes', 'normal'},
    'biological_sex': {'male', 'female', 'other'},
    'insulin_treatment': {'open_loop', 'closed_loop', 'hybrid_closed_loop', 'no_insulin'},
}
NUMERICAL_METADATA = ('age', 'bmi')
UNKNOWN_NUMERIC = -1.0


class InvalidCSV(ValueError):
    """Raised when an uploaded CSV cannot be used for a forecast."""


def _timestamp_column(df):
    for name in TIMESTAMP_COLUMNS:
        if name in df.columns:
            return name
    raise InvalidCSV(f'CSV must contain one of {list(TIMESTAMP_COLUMNS)}.')


def validate(df, expected_user_id=None):
    """Validate an uploaded CSV and return ``(dataframe, user_id, metadata)``.

    The returned dataframe is a sorted copy with a timezone-aware timestamp
    column; the caller's frame is never modified.

    Args:
        df: Raw ``pd.DataFrame`` as read from the upload.
        expected_user_id: If given, the CSV's ``user_id`` must equal it.

    Returns:
        tuple: ``(df, user_id, metadata)`` where ``metadata`` holds only the
        five model slots that were actually present.

    Raises:
        InvalidCSV: On any violation, with a client-safe message.
    """
    if df is None or len(df) == 0:
        raise InvalidCSV('CSV contains no rows.')

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise InvalidCSV(f'CSV must contain a {column!r} column.')

    df = df.copy()
    timestamp = _timestamp_column(df)

    user_id = _validate_user_id(df, expected_user_id)
    df = _validate_timestamps(df, timestamp)

    if len(df) < MIN_ROWS:
        raise InvalidCSV(f'Need at least {MIN_ROWS} rows ({MIN_ROWS * SAMPLING_INTERVAL_MIN} '
                         f'minutes) of history; got {len(df)}.')

    _validate_timeseries_values(df)
    metadata = _extract_metadata(df)
    return df, user_id, metadata


def _validate_user_id(df, expected_user_id):
    values = df['user_id'].dropna().unique()
    if len(values) == 0:
        raise InvalidCSV('Column user_id is empty.')
    if len(values) > 1:
        raise InvalidCSV(f'CSV must describe exactly one user; found {len(values)} user_id values.')
    if df['user_id'].isna().any():
        raise InvalidCSV('Column user_id has blank entries.')

    user_id = str(values[0]).strip()
    if not USER_ID_PATTERN.match(user_id):
        raise InvalidCSV(
            f'Invalid user_id {user_id!r}: expected 1-64 characters from [A-Za-z0-9_-]. '
            'It is used as a path segment, so it is validated before any computation.')
    if expected_user_id is not None and user_id != str(expected_user_id).strip():
        raise InvalidCSV('user_id in the CSV does not match the requested user_id.')
    return user_id


def _parse_timestamp(value):
    """Parse one value, preserving whether it carried a timezone."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT


def _validate_timestamps(df, timestamp):
    # Parse without normalising to UTC first: pd.to_datetime(..., utc=True) would
    # silently relabel a naive local timestamp as UTC, shifting forecast_origin by
    # the user's offset and returning a forecast for the wrong wall-clock time.
    parsed = df[timestamp].map(_parse_timestamp)
    if parsed.isna().any():
        raise InvalidCSV(f'Column {timestamp!r} contains unparseable or blank timestamps.')

    naive = parsed.map(lambda t: t.tzinfo is None)
    if naive.any():
        raise InvalidCSV(
            f'Column {timestamp!r} must be timezone-aware; {int(naive.sum())} of {len(parsed)} '
            'values have no UTC offset. Add an offset (e.g. "2026-09-10 00:00:00+00:00") — '
            'timezone-free timestamps are rejected rather than assumed to be UTC.')

    times = pd.to_datetime(parsed, utc=True)
    if times.duplicated().any():
        raise InvalidCSV('Timestamps must be unique.')

    df[timestamp] = times
    df = df.sort_values(timestamp).reset_index(drop=True)

    gaps = df[timestamp].diff().dropna()
    expected = pd.Timedelta(minutes=SAMPLING_INTERVAL_MIN)
    if not gaps.eq(expected).all():
        raise InvalidCSV(f'Timestamps must be exactly {SAMPLING_INTERVAL_MIN} minutes apart; '
                         'represent gaps as rows with blank measurements.')
    return df


def _validate_timeseries_values(df):
    for column in TIMESERIES_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors='coerce')
        if values.notna().sum() != df[column].notna().sum():
            raise InvalidCSV(f'Column {column!r} contains non-numeric values.')
        observed = values.dropna()
        if not np.isfinite(observed).all():
            raise InvalidCSV(f'Column {column!r} contains non-finite values.')
        if (observed < 0).any():
            raise InvalidCSV(f'Column {column!r} must be non-negative; use blanks for missing data.')

    _validate_usable_glucose(df)


def _validate_usable_glucose(df):
    """Require real glucose at the end of the history.

    A zero is NOT an observation: ``DataPrepper`` converts exact zeros to missing
    values before scaling, so an all-zero ``cbg`` column reaches the model as
    entirely missing and would still yield confident-looking numbers.  Usability is
    therefore counted after applying that same zero-to-missing rule.

    The tail is what matters: the forecast window ends at the last row, and the
    pipeline's own ``required_samples_window``/``required_samples_during_test``
    settings call for 24 valid glucose samples there.
    """
    values = pd.to_numeric(df['cbg'], errors='coerce')
    usable = values.notna() & (values != 0)

    if not usable.any():
        raise InvalidCSV('Column cbg has no usable glucose values; zeros are treated as '
                         'missing, so leave gaps blank rather than writing 0.')

    tail = usable.tail(MIN_ROWS)
    if not tail.all():
        raise InvalidCSV(
            f'The last {MIN_ROWS} rows must all contain usable glucose; '
            f'{int((~tail).sum())} of them are blank or zero. The forecast is made from the '
            'end of the history, and zeros count as missing.')


def _constant_value(df, column):
    """Return the single non-null value of a metadata column, or None if absent/blank."""
    if column not in df.columns:
        return None
    values = df[column].dropna().unique()
    if len(values) == 0:
        return None  # present but entirely blank -> treated as absent
    if len(values) > 1:
        raise InvalidCSV(f'Metadata column {column!r} must be constant for one user; '
                         f'found {len(values)} distinct values.')
    return values[0]


def _extract_metadata(df):
    metadata = {}

    for column, allowed in CATEGORICAL_METADATA.items():
        value = _constant_value(df, column)
        if value is None:
            continue
        value = str(value).strip()
        if value not in allowed:
            raise InvalidCSV(f'Metadata {column!r} has unsupported value {value!r}; '
                             f'expected one of {sorted(allowed)}.')
        metadata[column] = value

    for column in NUMERICAL_METADATA:
        value = _constant_value(df, column)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InvalidCSV(f'Metadata {column!r} must be numeric; got {value!r}.')
        if not np.isfinite(number):
            raise InvalidCSV(f'Metadata {column!r} must be a finite number.')
        # -1 is the model's unknown sentinel; any other negative is a mistake.
        if number < 0 and number != UNKNOWN_NUMERIC:
            raise InvalidCSV(f'Metadata {column!r} must be non-negative, or exactly '
                             f'{UNKNOWN_NUMERIC:g} to mean unknown; got {number:g}.')
        metadata[column] = number

    return metadata
