---
applyTo: "dataloaders/**"
---
# Dataloader Conventions

All dataset loaders inherit from `dataloaders.dataloader.Dataloader` (ABC) and must implement:

- `load_data()` — returns dict of `{participant_id: DataFrame}`
- `get_metadata()` — returns dict of participant metadata
- `_get_dataframe()` — parse a single raw file into a DataFrame
- `_get_dataset_specific_metadata()` — extract dataset-specific metadata fields

## DataFrame Schema

Every loader must produce DataFrames with at minimum these columns:
- `cbg` — continuous glucose monitoring values (mg/dL)
- `5minute_intervals_timestamp` — datetime index at 5-minute resolution

Optional standard columns: `basal`, `bolus`, `carbInput`, `iob`, `cob`, `hr`, `gsr`

## Registration

After creating a new dataloader:
1. Add import to `dataloaders/__init__.py`
2. Add the class name to the `globals()` lookup in `data/handler.py` (`create_dataloader_instance`)
3. Add a dataset config entry in `run_multiple.py` / `run_multiple2.py` `dataset_configs` dict

## Caching

`DataHandler` caches loaded data as pickles in `standardized_datasets/{dataset_name}/`. Set `load_from_pkl=False` to force reload from raw files.
