# Blood Glucose Forecasting API — Backend

**Status: implemented.**
Serves a forecast from one user's CGM CSV through the existing pipeline entry point.
Developed and tested against the MELISSA sample CSVs in
`standardized_datasets/MELISSA_user001/test/` only (12 files, 30 to 2592 rows, `user_id`
`MELISSA_user001`).

---

## 1. Decisions

| # | Decision |
|---|---|
| D1 | Short histories supported (less than the 8-day window). |
| D2 | Forecast only — no backtest, no metrics on the request path. |
| D3 | Keep the full sequence; do **not** drop the last `forecast_steps` from the input. |
| D4 | Discard no batch — every window fed returns a prediction. |
| D5 | Existing training/evaluation behaviour preserved. All changes additive. |
| D6 | Dataloader name: `DataloaderMelissa`. |
| D7 | Synchronous endpoints (110 ms p50 warm on CPU). |

---

## 2. Constraints that must not be broken

Each was reproduced against this repo. Breaking one produces a *silently wrong* forecast, not an error.

| constraint | where | consequence if ignored |
|---|---|---|
| `Scaler` **fits and saves** when its `.pkl` is missing | [scaler.py:41-48](data/scaler.py#L41-L48) | a missing artifact fits a scaler on one request's rows *and overwrites the bundle*. Prevented by passing `ForecastOnlyScaler` (§4), which raises inside `fit_and_save` **before** `joblib.dump`. Checking file hashes up front cannot prevent this — the scalers are opened later in the request |
| `trainer.test()` drops its last batch (`preds[:-1]`) | [exp_long_term_forecasting.py:1155](trainers/exp_long_term_forecasting.py#L1155) | the forecast window is the only one → returns `shape=(0,1)` |
| `SequenceDataset` truncates the last `forecast_steps` from the input | [dataset.py:222-240](data/dataset.py#L222-L240) | no window can forecast past the last observation (D3) |
| Variance is trained against the **`mean`** head, not `forecast` | [exp_long_term_forecasting.py:944-946](trainers/exp_long_term_forecasting.py#L944-L946) | intervals around `predicted_glucose_mgdl` describe a quantity the variance never fitted |
| Training clamps variance; inference does not | [632](trainers/exp_long_term_forecasting.py#L632) vs [1112](trainers/exp_long_term_forecasting.py#L1112) | `sqrt` of negative variance → `NaN` |
| Alarm heads: one score per window, **fixed 60 min**, hybrid target | [exp_long_term_forecasting.py:642-684](trainers/exp_long_term_forecasting.py#L642-L684) | not per-step, not rescalable to the request horizon, not a calibrated probability |
| Model reads exactly 5 metadata slots by **exact key name** | [iTransformerMasked.py:193-247](architectures/iTransformerMasked.py#L193-L247) | a wrong key silently becomes `unknown_*`. `age`/`bmi` use **`-1`** for unknown |
| `DataPrepper` converts **exact zeros to missing** | [prepper.py:84-90](data/prepper.py#L84-L90) | a zero is not an observation; an all-zero `cbg` column reaches the model as entirely missing yet still returns numbers |
| `pd.to_datetime(..., utc=True)` relabels naive timestamps | — | a naive local time is read as UTC, shifting `forecast_origin` by up to 14 h, silently |
| `pd.read_csv` infers `user_id` dtype | — | a valid id such as `"0001"` becomes the integer `1` and fails the id comparison |
| `DataPrepper` mutates the frame it is given | [prepper.py:84-90](data/prepper.py#L84-L90) | cross-request corruption unless a copy is passed |
| Mask/padding length mismatch | [prepper.py:144-163](data/prepper.py#L144-L163) | **preserved deliberately** (D5) — do not "fix" |
| Centred Gaussian smoothing on `bolus`/`carbInput` | [prepper.py:134-138](data/prepper.py#L134-L138) | offline evaluation sees ~30 min of post-origin data a live request cannot → train/serve skew |

**Metadata slots:** `diagnosis_type` (`type1`/`type2`/`prediabetes`/`normal`), `biological_sex`
(`male`/`female`/`other`), `insulin_treatment` (`open_loop`/`closed_loop`/`hybrid_closed_loop`/`no_insulin`),
`age`, `bmi`. All five come from CSV columns.

**Model bundle:** `inference/test/` holds **AI4Food-trained weights** — development only, not
clinically meaningful. The original MELISSA download arrived as all-null bytes; re-transfer as a
single archive and verify SHA-256 at the source.

---

## 3. API contract

`POST /v1/forecasts` — multipart: `user_id`, `file` (CSV), `horizon_minutes` ∈ {30, 60, 120}.

**CSV:** `timestamp` (unique, 5-min spacing, **timezone-aware — naive rejected**), `cbg`, `user_id`
(must match the request, read as a string, validated `^[A-Za-z0-9_-]{1,64}$`). Optional: `bolus`,
`carbInput`, `basal`, and the five metadata columns (constant per file; unrecognised values are
rejected, not degraded). Minimum 24 rows, and the **last 24 rows must all carry usable glucose** —
blank or zero does not count. Shorter histories are left-padded. Malformed files (empty, ragged,
non-UTF-8) raise `InvalidCSV`, never a raw pandas exception.

**Response:** two separate objects, each with its own horizon.

```jsonc
"forecast":   { "horizon_minutes": 30,          // as requested; 30/60/120 -> exactly 6/12/24 steps
                "interval": { "coverage": "nominal_95", "centred_on": "probabilistic_mean",
                              "centred_on_predicted_glucose": false, "calibrated": false },
                "steps": [ { "prediction_time", "minutes_ahead",
                             "predicted_glucose_mgdl",   // deterministic head — the prediction
                             "pred_std_mgdl", "ci_low_mgdl", "ci_high_mgdl",
                             "glycemic_zone" } ] }

"event_risk": { "horizon_minutes": 60, "fixed": true,    // ignores the requested horizon
                "scale": "uncalibrated_score",
                "thresholds_mgdl": { "hypo": 70, "hyper": 180 },
                "hypo": 0.005, "hyper": 0.033 }
```

Per-step CSV at `GET /v1/forecasts/{run_id}.csv` carries **no event-risk column** — repeating a
window-level score per row is the misreading the split exists to prevent. The score is saved
separately as `event_risk.csv`, one row per run, joinable to the per-step file on `run_id`.

`model_version` is derived from the verified artifact digests (e.g. `test@992041340037`), not the
directory name, so replacing a bundle in place changes it.

`user_id` is validated before any computation, then used as one path segment with a containment
assert. It is never a model input.

**Every run is saved** under `forecast_runs/<user_id>/<run_id>/` by default (`forecast.csv`,
`event_risk.csv`, `forecast.json`), which also backs `GET /v1/forecasts/{run_id}.csv`. `run_id` is a UTC timestamp
plus a random suffix, so concurrent requests cannot collide. Set `output_dir=None` to keep results
in memory. **There is no retention or cleanup policy**, and runs hold per-user glucose data.

---

## 4. What was built

| file | role |
|---|---|
| [dataloaders/dataloader_melissa.py](dataloaders/dataloader_melissa.py) | `DataloaderMelissa` |
| [dataloaders/csv_validation.py](dataloaders/csv_validation.py) | typed `InvalidCSV` validation |
| [service/bundle.py](service/bundle.py) | `verify_model_bundle`, `assert_bundle_intact`, `bundle_version` |
| [service/forecast_input.py](service/forecast_input.py) | window construction, `ForecastOnlyScaler` |
| [service/forecast_model.py](service/forecast_model.py) | forward pass |
| [service/result.py](service/result.py) | response shape, CSV/JSON |
| [service/forecast_run.py](service/forecast_run.py) | orchestration, safe run dirs |
| [service/forecaster.py](service/forecaster.py) | warm `Forecaster` |
| [service/\_\_main\_\_.py](service/__main__.py) | CLI |
| [pipeline/config.py](pipeline/config.py) | per-request config, never written to disk |

**Forecast path:** CSV → `DataloaderMelissa` → `DataPrepper` **unmodified** → slice
`input_data[-2304:]` → `PeriodicityReshape` → `model(...)` → inverse-transform.

**Why this shape:** calling `DataPrepper` unmodified and slicing its output gives byte-identical
preprocessing while satisfying D3 — so no shared code changes and D5 needs no regression baseline.
Calling the model directly instead of `trainer.test` satisfies D4 with `test()` untouched.

**Preventing a refit without touching shared code:** `ForecastOnlyScaler(StandardScaler)` overrides
`fit`/`partial_fit` to raise, and separate instances are passed as `scaler_class_x`/`scaler_class_y`.
`Scaler.fit_and_save` calls `fit` before `joblib.dump`, so a missing artifact fails before anything
is written; on the normal path `load_scaler` rebinds to the loaded scaler and the guard is discarded.
The parameter comparison that follows is a *consistency* check against swapped artifacts — matching
means and scales would not, on its own, prove nothing was fitted.

**Accepted costs:** `DataPrepper` builds a `SequenceDataset` the path discards (~36-162 ms), and
re-reads both scalers from disk per request — the warm runtime holds the **model only**.

### Running it

```bash
python -m service --csv <file> --user-id <id> --horizon 30    # from the repo root
PYTHONPATH=/path/to/BG_forecasting python -m service --csv ... --user-id ...   # from anywhere
```

stdout is clean CSV; progress, warnings and event risk go to stderr. `--format json` emits the full
payload. In Python, build `Forecaster()` once and reuse it.

### Footprint on existing code — 15 lines

```
 data/handler.py          | 1 +   # register DataloaderMelissa
 dataloaders/__init__.py  | 1 +   # register DataloaderMelissa
 pipeline/orchestrator.py | 13 +  # forecast dispatch branch
```

`main(config, train=True, test=True, *, runtime=None)` — `runtime` keyword-only, service imported
inside the branch, returns before `init_experiment_directory` (which would otherwise write into
the shared model dir on every request).

### Verified behaviour

Each of these was reproduced and confirmed against the MELISSA CSV and the bundle in
`inference/test`.

- **A refit is prevented, not merely detected.** Deleting the scaler mid-request (after
  verification, before `DataPrepper`) makes the request raise; the artifact is **not** recreated
  and no other artifact is modified.
- **Concurrent requests are isolated.** Two different patients — distinct histories, metadata and
  horizons — run interleaved through one `Forecaster` and each returns exactly what it returns when
  run alone. Neither `base_config` nor the model `state_dict` changes.
- A missing, all-null, or content-swapped bundle artifact raises. A reordered `features` list raises
  via the scalers' `feature_names_in_`.
- All-zero or blank glucose in the last 24 rows is rejected; `"0001"` keeps its leading zeros;
  `"../escape"` is rejected even when nothing is written; empty, ragged and binary CSVs raise
  `InvalidCSV`.
- A config without `mode` still reaches `init_experiment_directory`; forecast mode returns before
  it. Importing `pipeline.orchestrator` loads no `service` module. All 11 dataloaders resolve, and
  Colas keeps `id_column='id'` and its `diagonosis_type → T2DM` mapping.

---

## 5. Outstanding

- Choose a web framework and add it to [requirements.txt](requirements.txt). Endpoints are a thin
  layer over `Forecaster.forecast(user_id, csv_path, horizon_minutes)`.
- **No automated test suite.** The behaviour above was verified manually; there is nothing guarding
  it against regressions.
- Deferred: `GET /v1/forecasts/{run_id}/event-risk` (a read of a stored run — never a recompute;
  must carry `run_id`, `forecast_origin`, `model_version`).
- **No retention or cleanup for saved runs**, which are now written by default. Runs accumulate
  indefinitely and contain per-user glucose data; decide on a policy before any real volume.
- A read-only bundle directory remains useful deployment hardening, though the service no longer
  depends on it.
- **Unmeasured:** history length 2304 vs 2280 (288 vs 285 tokens; differed by up to 6.46 mg/dL in
  one synthetic comparison), and the centred-smoothing skew — requires offline runs with history
  truncated at each origin. Interval widths remain **unattributed**: AI4Food weights, 24-row
  history and the token-count change are confounded.
- Swap in the MELISSA bundle when re-obtained — one `model_dir` change.
