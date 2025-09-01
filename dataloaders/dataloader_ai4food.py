import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob
import json

class DataloaderAI4Food(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        self.train_test_divide = 84962
        #self.train_test_divide_T2DM = 2090 #participants with a number higher than this will be put in the test set
        #self.train_test_divide_T1DM = 1011 #participants with a number higher than this will be put in the test set

    def load_data(self):
        #train_path = os.path.join(self.directory_path, "Shanghai")
        files = glob(self.directory_path + os.sep + "DS4_Biomarkers" + os.sep + "glucose_levels" + os.sep + "*.csv")

        for file in files:
            if os.path.basename(file).startswith('~$'):
                continue  # skip temp files
            id = int(file.split(os.sep)[-1].split('_')[1])
            df, patient_id = self._get_dataframe(file)
            if id < self.train_test_divide:
                self.train_dataframes[patient_id] = df
            else:
                self.test_dataframes[patient_id] = df
            self.all_dataframes[patient_id] = df


        #return self.all_dataframes, self.train_dataframes, self.test_dataframes
    
    def _get_dataset_specific_metadata(self):
        """Extract patient metadata from an Excel file into a dictionary."""
        filename = os.path.join(self.directory_path, 'participant_information.csv')
        df = pd.read_csv(filename)
        
        data_mapping = {"sex": "biological_sex", "age": "age"}
        df = df.rename(columns=data_mapping)
        df["bmi"] = df["usual_weight_kg"] / (df["height_cm"] / 100) ** 2
        # Rename columns according to mapping
        og_metadata = df.set_index("id").to_dict(orient="index")
        # Map biological_sex values
        for patient, meta in og_metadata.items():
            meta["insulin_treatment"] = "no_insulin"
            meta["diagonosis_type"] = "normal"
            if "biological_sex" in meta:
                if meta["biological_sex"] == "Female":
                    meta["biological_sex"] = "female"
                elif meta["biological_sex"] == "Male":
                    meta["biological_sex"] = "male"
        
        #self.train_metadata = {}
        #self.test_metadata = {}

        for patient, meta in og_metadata.items():
            
            try:
                patient_num = int(patient.split(os.sep)[-1].split('_')[-1])
            except Exception:
                continue
            if patient_num < self.train_test_divide:
                
                self.train_metadata[patient] = meta
            else:
                self.test_metadata[patient] = meta


    def _get_dataframe(self, file, calculate_iob=True):
        import os
        import numpy as np
        import pandas as pd

        # Load
        df = pd.read_csv(file)

        # Standardize column names we care about
        df = df.rename(columns={
            "glucose_value_in_mg_dl": "cbg",
            "timestamp": "5minute_intervals_timestamp"
        })

        # Ensure expected columns exist so resampling keeps them
        for col in ["carbInput", "hr", "basal", "bolus"]:
            if col not in df.columns:
                df[col] = np.nan

        # --- Upsample/resample to 5-minute frequency with timestamp rounding ---
        time_col = "5minute_intervals_timestamp"
        if time_col in df.columns:
            # Parse timestamps (keep tz if present)
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

            # 1) Round original timestamps to nearest 5 minutes to avoid dropping off-grid samples
            rounded = df.copy()
            rounded.index = rounded.index.round("5T")

            # If multiple rows round to the same stamp, keep the first (or change to .mean() if desired)
            rounded = rounded[~rounded.index.duplicated(keep="first")]

            # 2) Build full 5-min grid (preserve timezone if any)
            grid = pd.date_range(
                start=rounded.index.min(),
                end=rounded.index.max(),
                freq="5T",
                tz=rounded.index.tz
            )

            # 3) Reindex to the full grid (no fill yet)
            df = rounded.reindex(grid)
            df.index.name = time_col

            # 4) Coerce numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 5) Treat insulin columns as spikes: fill NaNs with 0 only for these
            insulin_cols = [c for c in ["bolus", "basal", "Insulin dose - s.c.", "Insulin dose - i.v."] if c in df.columns]
            if insulin_cols:
                df[insulin_cols] = df[insulin_cols].fillna(0)

            # 6) Interpolate ONLY short internal gaps (≤ 15 minutes → up to 3 steps on a 5T grid)
            limit_steps = int(pd.Timedelta("15min") / pd.Timedelta("5min"))  # 3
            continuous_cols = [c for c in df.select_dtypes(include="number").columns if c not in insulin_cols]

            if continuous_cols:
                df[continuous_cols] = df[continuous_cols].interpolate(
                    method="time",
                    limit=limit_steps,
                    limit_direction="both",
                    limit_area="inside"  # no extrapolation beyond observed range
                )

            # IMPORTANT: do NOT ffill/bfill globally (would inappropriately fill long gaps)
            df = df.reset_index()

        # Add iob/cob placeholders if requested (calculation may happen later)
        if calculate_iob:
            if "iob" not in df.columns:
                df["iob"] = np.nan
            if "cob" not in df.columns:
                df["cob"] = np.nan

        patient_id = file.split(os.sep)[-1].split(".")[0].replace("_glucose_levels", "")
        return df, patient_id



