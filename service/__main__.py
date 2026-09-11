"""Command-line forecast.

From the repository root::

    python -m service --csv path/to/history.csv --user-id user_001 --horizon 30

From any other directory, put the repository root on the path -- ``python -m``
resolves the package before this module can adjust ``sys.path``::

    PYTHONPATH=/home/maria/code/BG_forecasting python -m service --csv ... --user-id ...

Writes CSV to stdout by default so it can be redirected or piped.  Use
``--format json`` for the full payload including the event-risk object, which the
per-step CSV deliberately omits (it is a single window-level score, not a per-step
value).
"""

import argparse
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, '.matplotlib'))

DEFAULT_MODEL_DIR = os.path.join(ROOT, 'inference', 'test')
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, 'forecast_runs')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m service',
        description='Forecast blood glucose from one user\'s CGM CSV.')
    parser.add_argument('--csv', required=True,
                        help="Path to the user's CGM CSV.")
    parser.add_argument('--user-id', required=True,
                        help="Must match the CSV's user_id column.")
    parser.add_argument('--horizon', type=int, default=30, choices=[30, 60, 120],
                        help='Forecast horizon in minutes (default: 30). '
                             'Returns 6, 12 or 24 five-minute steps.')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                        help='Format for --stdout: csv = per-step rows (default); '
                             'json = full payload including the 60-minute event risk. '
                             'Both are always saved to the run directory.')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help='Directory holding best_model.pth, both scalers and '
                             'model_config.json (default: inference/test).')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Where runs are saved, as <output-dir>/<user_id>/<run_id>/ '
                             f'(default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save anything; print the result instead.')
    parser.add_argument('--stdout', action='store_true',
                        help='Also print the result, for piping or redirection.')
    parser.add_argument('--device', default='cpu',
                        help="'cpu' (default) or 'cuda'. CPU serves in ~110 ms and "
                             'avoids CUDA warm-up.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show progress and library output on stderr.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')

    # Imported after sys.path is set up.
    from dataloaders.csv_validation import InvalidCSV
    from service.bundle import BundleError
    from service.forecast_run import ForecastError
    from service.forecaster import Forecaster

    output_dir = None if args.no_save else args.output_dir

    try:
        # The pipeline prints progress to stdout; keep stdout clean for the result.
        stdout, sys.stdout = sys.stdout, sys.stderr
        try:
            forecaster = Forecaster(model_dir=args.model_dir, device=args.device,
                                    output_dir=output_dir)
            result = forecaster.forecast(args.user_id, args.csv, horizon_minutes=args.horizon)
        finally:
            sys.stdout = stdout
    except InvalidCSV as exc:
        sys.exit(f'Invalid CSV: {exc}')
    except BundleError as exc:
        sys.exit(f'Model bundle problem: {exc}')
    except ForecastError as exc:
        sys.exit(f'Cannot forecast: {exc}')
    except FileNotFoundError as exc:
        sys.exit(f'File not found: {exc}')

    for warning in result.warnings:
        print(f'warning: {warning}', file=sys.stderr)

    risk = result.to_dict()['event_risk']
    # Event risk is window-level and fixed at 60 minutes, so it is never a CSV column;
    # report it on stderr so a redirected stdout still yields clean CSV.
    print(f'event risk (next 60 min, uncalibrated): '
          f"hypo={risk['hypo']:.4f} hyper={risk['hyper']:.4f}", file=sys.stderr)

    # --stdout is for piping; without it stdout names where the run was saved, which
    # keeps that scriptable too.
    if args.stdout or output_dir is None:
        if args.format == 'json':
            print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
        else:
            print(result.to_csv(), end='')

    if output_dir is not None:
        directory = os.path.join(output_dir, result.user_id, result.run_id)
        if not args.stdout:
            print(directory)
        for name in ('forecast.csv', 'event_risk.csv', 'forecast.json'):
            print(f'saved {os.path.join(directory, name)}', file=sys.stderr)


if __name__ == '__main__':
    main()
