# Forecast service

Predicts blood glucose for one user from a CGM CSV, using a trained
iTransformerMasked checkpoint. Returns a per-step trajectory with uncertainty, plus
a separate hypo/hyper event risk.

This package is **additive**: it calls the training/evaluation pipeline without
changing its calculations. Forecast dispatch returns early in `pipeline.orchestrator.main`;
evaluation exports load lazily so forecasting does not require `cg_ega`. Design rationale lives in
[API_IMPLEMENTATION_PLAN.md](API_IMPLEMENTATION_PLAN.md).

---

## Setup

```bash
source ../.venv/bin/activate        # the project's existing environment, from BG_forecasting/
```

(If your checkout uses a conda env instead of `.venv`, activate that — the service has no
dependencies beyond what the training/evaluation pipeline already requires.)

No web framework is required — there is no HTTP server yet, only a library and a CLI.

---

## Quickstart

Run from the `BG_forecasting/` repo root. The sample bundle lives nested under the dataset
directory, not at the `--model-dir` default (see the [CLI](#cli) table), so pass it explicitly:

```bash
python -m service \
  --csv standardized_datasets/MELISSA_user001/test/cgm_20260917T235500Z.csv \
  --user-id MELISSA_user001 --horizon 30 \
  --model-dir standardized_datasets/MELISSA_user001/inference/test
```

```
user_id,forecast_origin,prediction_time,minutes_ahead,predicted_glucose_mgdl,pred_std_mgdl,ci_low_mgdl,ci_high_mgdl,glycemic_zone
MELISSA_user001,2026-09-17T23:55:00+00:00,2026-09-18T00:00:00+00:00,5,123.54,16.98,91.55,158.10,in_range
MELISSA_user001,2026-09-17T23:55:00+00:00,2026-09-18T00:05:00+00:00,10,122.78,19.03,85.69,160.30,in_range
...
```

```
<repo-root>/BG_forecasting/forecast_runs/MELISSA_user001/20260911T123543_c6a7af719302
```

**Every run is saved.** stdout is the run directory; the forecast itself is written
inside it as `forecast.csv` and `forecast.json`. Add `--stdout` to print the forecast
too (for piping), or `--no-save` to print it without writing anything.

Sample data lives in `standardized_datasets/MELISSA_user001/test/`, one file per
history length — from `cgm_20260910T022500Z.csv` (30 rows, 2.5 h) up to
`cgm_20260918T235500Z.csv` (2592 rows, 9 days). All use `user_id` `MELISSA_user001`.

From **any** directory, put the repo root on the path — `python -m` resolves the
package before the module can fix `sys.path` itself:

```bash
PYTHONPATH=<repo-root>/BG_forecasting python -m service --csv history.csv --user-id MELISSA_user001
```

Progress, warnings and the event risk always go to stderr, so with `--stdout` or
`--no-save` a redirect (`> out.csv`) yields a directly loadable file.

---

## CLI

| flag | meaning |
|---|---|
| `--csv` | path to the user's CGM CSV (required) |
| `--user-id` | must match the CSV's `user_id` column (required) |
| `--horizon` | 30 / 60 / 120 minutes → 6 / 12 / 24 steps (default 30) |
| `--format` | format used by `--stdout`: `csv` (default) or `json`. Both are always saved |
| `--model-dir` | bundle directory, relative to the repo root (default `inference/test`; the MELISSA sample bundle is nested at `standardized_datasets/MELISSA_user001/inference/test`, not the default — pass it explicitly) |
| `--output-dir` | where runs are saved (default `forecast_runs/`) |
| `--no-save` | write nothing; print the forecast instead |
| `--stdout` | also print the forecast, for piping |
| `--device` | `cpu` (default) or `cuda` |
| `-v` | show progress on stderr |

---

## Python API

Build `Forecaster` **once** per process and reuse it — construction verifies the
bundle and loads the model (~0.3 s); each forecast is then ~110 ms.

```python
from service.forecaster import Forecaster

forecaster = Forecaster()                     # or Forecaster(model_dir=..., device='cuda')
result = forecaster.forecast(
    user_id='MELISSA_user001',
    csv_path='history.csv',
    horizon_minutes=30,
)

result.steps              # list of per-step dicts
result.to_csv()           # per-step CSV (no event-risk column — see below)
result.to_dict()          # full JSON payload
result.event_risk_hypo    # window-level score, fixed 60-minute horizon
result.warnings           # e.g. short history was padded
```

Nothing is written to disk unless `Forecaster(output_dir=...)` is given.

An HTTP layer is a thin wrapper over `forecast(...)`; the exceptions below map
directly onto 4xx/5xx.

### Through the pipeline entry point

```python
from pipeline.config import build_forecast_config
from pipeline.orchestrator import main
from service.bundle import load_config

# model_dir here is the MELISSA sample bundle's actual (nested) location -- see the
# --model-dir row in the CLI table above.
model_dir = 'standardized_datasets/MELISSA_user001/inference/test'
config = build_forecast_config(load_config(model_dir), model_dir=model_dir,
                               csv_path='history.csv', user_id='MELISSA_user001',
                               horizon_minutes=60, output_dir='forecast_runs')
result = main(config)
```

---

## Input CSV

One user per file. Columns:

| column | required | rules |
|---|---|---|
| `user_id` | yes | constant, must equal the requested id, `[A-Za-z0-9_-]{1,64}`. Read as text, so `0001` stays `0001` |
| `timestamp` | yes | unique, exactly 5 minutes apart, **timezone-aware** |
| `cbg` | yes | mg/dL, non-negative or blank |
| `bolus`, `carbInput` | no | units / grams; blank where unknown |
| `basal` | no | carried through, **not** a model input |
| `diagnosis_type` | no | `type1`, `type2`, `prediabetes`, `normal` |
| `biological_sex` | no | `male`, `female`, `other` |
| `insulin_treatment` | no | `open_loop`, `closed_loop`, `hybrid_closed_loop`, `no_insulin` |
| `age`, `bmi` | no | numeric; **`-1` means unknown** |

Metadata columns must be constant within the file. Omitted or blank metadata falls
back to the model's `unknown_*` tokens; an *unrecognised* value is rejected rather
than silently degraded.

**Minimum 24 rows**, and the **last 24 rows must all carry usable glucose**. Blank
and zero both count as missing — the pipeline converts exact zeros to missing
values, so write blanks for gaps, never `0`.

The service uses the latest `feature_window - forecast_steps` samples (2280 for
this bundle), matching training while retaining the newest observation. Shorter
histories are left-padded with the model's missing-value token. Matching training
length changes predictions; improved accuracy has not been established.

```csv
user_id,timestamp,cbg,diagnosis_type,biological_sex,age,bmi,bolus,basal,carbInput,insulin_treatment
MELISSA_user001,2026-09-10 00:00:00+00:00,120,type1,female,42,24.1,0,0.8,0,open_loop
```

---

## Output

### Where it goes

**Every run is saved**, under `forecast_runs/<user_id>/<run_id>/`:

```
forecast_runs/MELISSA_user001/20260911T123543_c6a7af719302/forecast.csv     # per-step trajectory
forecast_runs/MELISSA_user001/20260911T123543_c6a7af719302/event_risk.csv   # one row per run
forecast_runs/MELISSA_user001/20260911T123543_c6a7af719302/forecast.json    # everything
```

Both CSVs carry `run_id`, so runs stack and join:

```python
steps = pd.concat(pd.read_csv(f) for f in glob.glob('forecast_runs/*/*/forecast.csv'))
risk  = pd.concat(pd.read_csv(f) for f in glob.glob('forecast_runs/*/*/event_risk.csv'))
steps.merge(risk, on='run_id')
```

`<run_id>` is a UTC timestamp plus a random suffix, so no run overwrites another —
including two requests arriving in the same second. stdout is the run directory, so
it can be captured:

```bash
DIR=$(python -m service --csv ... --user-id MELISSA_user001)
cat "$DIR/forecast.csv"
```

| variation | effect |
|---|---|
| `--output-dir runs` | save somewhere else |
| `--stdout` | save **and** print the forecast (for piping) |
| `--no-save` | print only, write nothing |

In Python, `Forecaster()` saves to the same default; `Forecaster(output_dir=None)`
keeps results in memory.

`forecast_runs/` is in `.gitignore`. **Runs accumulate indefinitely — there is no
cleanup policy**, and they contain per-user glucose data, so decide on retention
before running this at any volume.

### Per-step CSV — one row per 5-minute step

```
run_id,user_id,forecast_origin,prediction_time,minutes_ahead,predicted_glucose_mgdl,pred_std_mgdl,ci_low_mgdl,ci_high_mgdl,glycemic_zone
20260911T124401_884eb4b683d2,MELISSA_user001,2026-09-17T23:55:00+00:00,2026-09-18T00:00:00+00:00,5,123.54,16.98,91.55,158.10,in_range
```

| column | example | meaning |
|---|---|---|
| `run_id` | `20260911T124401_…` | identifies this run; joins to `event_risk.csv` |
| `user_id` | `MELISSA_user001` | echoed from the request |
| `forecast_origin` | `2026-09-17T23:55Z` | timestamp of the **last observed** reading; the forecast begins after it. Same on every row |
| `prediction_time` | `2026-09-18T00:00Z` | the UTC instant this row predicts |
| `minutes_ahead` | `5` | minutes past the origin — 5, 10, 15 … |
| `predicted_glucose_mgdl` | `123.54` | **the prediction**, mg/dL |
| `pred_std_mgdl` | `16.98` | the model's uncertainty, σ, in mg/dL. Larger = less confident |
| `ci_low_mgdl` / `ci_high_mgdl` | `91.55` / `158.10` | plausible range, ±1.96σ |
| `glycemic_zone` | `in_range` | `low` (<70), `in_range` (70-180), `high` (>180), from `predicted_glucose_mgdl` |

Row counts are fixed by the horizon: **30 → 6 rows, 60 → 12, 120 → 24.**

#### How to read the interval

Two caveats, both of which matter:

- **It is "nominal" 95%, not verified 95%.** Calibration has never been checked, so
  do not rely on the coverage figure.
- **It is not centred on `predicted_glucose_mgdl`.** The model has two output heads:
  a deterministic one (reported as `predicted_glucose_mgdl`, trained with MSE) and a
  probabilistic one trained jointly with the variance. σ describes the *probabilistic*
  head, so the interval is built around that, and is therefore slightly asymmetric
  about the reported prediction — above, the midpoint is 124.8 while the prediction is
  123.5. That is intended. Centring the interval on the point forecast would attach it
  to a quantity the variance was never fitted against.

  The probabilistic mean itself is not reported; only its interval is.

### JSON (`--format json`)

```jsonc
{
  "run_id": "20260911T121947_de791db36f4e",
  "user_id": "MELISSA_user001",
  "forecast_origin": "2026-09-17T23:55:00+00:00",
  "model_version": "test@992041340037",   // from artifact checksums, not the dir name
  "forecast": {
    "horizon_minutes": 30, "unit": "mg/dL", "sampling_interval_minutes": 5,
    "interval": { "coverage": "nominal_95", "centred_on": "probabilistic_mean",
                  "centred_on_predicted_glucose": false, "calibrated": false },
    "steps": [ { "prediction_time": "...", "minutes_ahead": 5,
                 "predicted_glucose_mgdl": 123.54, "pred_std_mgdl": 16.98,
                 "ci_low_mgdl": 91.55, "ci_high_mgdl": 158.10,
                 "glycemic_zone": "in_range" } ]
  },
  "event_risk": { "horizon_minutes": 60, "thresholds_mgdl": { ... }, "hypo": ..., "hyper": ... },
  "warnings": []                          // e.g. a short history was padded
}
```

`model_version` is derived from the bundle's artifact checksums, so replacing a
bundle in place changes it — a forecast can always be traced to the weights that
produced it.

### Event risk — read this before using it

`event_risk` is **not** a per-step probability and is **not** rescalable:

- It is **one score per request**, over a **fixed 60-minute** horizon, regardless of
  the `--horizon` you asked for.
- It was trained against a hybrid target — 1.0 if any 3 consecutive samples lie
  beyond the threshold, otherwise the mean fraction of samples beyond it — so it is
  an **uncalibrated score**, not a probability.

For those reasons it is never a column in the per-step CSV. It is saved as its own
single-row **`event_risk.csv`**:

```
run_id,user_id,forecast_origin,model_version,event_horizon_minutes,scale,hypo_threshold_mgdl,hyper_threshold_mgdl,event_risk_hypo,event_risk_hyper
20260911T124401_884eb4b683d2,MELISSA_user001,2026-09-17T23:55:00+00:00,test@992041340037,60,uncalibrated_score,70,180,0.000188,0.000487
```

One row per run, so these stack into a table across many runs and join to the
per-step file on `run_id`. `event_horizon_minutes` is always 60. It is also in
`forecast.json`, and printed on stderr:

```
event risk (next 60 min, uncalibrated): hypo=0.0002 hyper=0.0005
```

Read `hypo=0.0002` as "very low risk", not as "a 0.02% chance". Use these for
ranking or thresholding, never as literal probabilities.

---

## Errors

| exception | meaning | suggested HTTP status |
|---|---|---|
| `InvalidCSV` | malformed, unparseable or non-conforming input | 400 |
| `ForecastError` | bad `user_id` format, unsupported horizon | 400 |
| `BundleError` | model artifacts missing, corrupt, or changed at runtime | 500 |

All carry messages safe to return to a caller. The CLI prints them without a
traceback and exits non-zero:

```
Invalid CSV: Column 'timestamp' must be timezone-aware; 24 of 24 values have no UTC offset...
Invalid CSV: The last 24 rows must all contain usable glucose; 1 of them are blank or zero...
Model bundle problem: Model artifact missing: .../scaler_input.pkl ...
```

---

## Model bundle

A bundle directory holds `best_model.pth`, `scaler_input.pkl`, `scaler_target.pkl`
and `model_config.json`. It is verified at startup and re-checked per request:
files present, non-empty, not all-null, loadable, the scalers' feature **order**
matching the config, and `load_state_dict(..., strict=True)` succeeding.

Treat a bundle as **immutable** while the service runs. A missing scaler would
otherwise cause the pipeline's `Scaler` to fit a new one on the request's own rows —
this is blocked by passing a scaler that refuses to fit, so such a request fails
loudly instead of returning a plausible wrong answer.

> The bundle currently in `inference/test/` contains **AI4Food-trained weights**.
> It is fine for development, but its predictions are not clinically meaningful.
> Point `--model-dir` at the intended model before any real use.

---

## Verified behaviour

Confirmed manually against the MELISSA CSV and the `inference/test` bundle:

- Deleting a scaler mid-request makes the request fail; no scaler is fitted and no
  artifact is overwritten.
- Two different patients running concurrently through one `Forecaster` each get the
  same result as when run alone; the shared model and config are unchanged.
- A missing, all-null or swapped bundle artifact raises, as does a reordered
  `features` list.
- `30/60/120` return exactly `6/12/24` steps, the first at origin + 5 min.
- The legacy training and evaluation path is unaffected: a config without `mode`
  behaves exactly as before, and importing `pipeline.orchestrator` loads no service
  module.

> **There is no automated test suite.** The above was checked by hand, so nothing
> guards it against regressions — worth adding before this is relied upon.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'service'`** — run `python -m service` from
the repository root, or set `PYTHONPATH` to the repo root.

**`KeyError: 0` / joblib errors** — a corrupt bundle. Check for all-null artifacts:
`tr -d '\0' < best_model.pth | wc -c` should be non-zero.

**Intervals look very wide** — expected with a short history and the AI4Food
development weights; the cause has not been isolated.

**First CUDA call takes ~40 s** — one-off warm-up. The CPU default avoids it and is
fast enough (~110 ms per request).
