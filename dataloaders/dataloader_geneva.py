import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob
import json

class DataloaderGeneva(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        self.id_column = "id"
        self.data_mapping =  {"5minute_intervals_timestamp": "Timestamp", "cbg": "Glucose [mmol/L]", "basal": None, "bolus": None, "hr": None, "carbInput": None}
    # create reverse mapping: csv column name -> internal column name
        self.reverse_data_mapping = {v: k for k, v in self.data_mapping.items() if v is not None}
        #self.metadata_mapping = {"age": "age","biological_sex": "gender", "bmi": "BMI", "diagonosis_type": "T2DM", "insulin_treatment": None}
        #self.reverse_metadata_mapping = {v: k for k, v in self.metadata_mapping.items() if v is not None}



    def load_data(self):
        # Iterate over CSV/Excel files in the provided directory_path and load each file
        patterns = ['*.csv', '*.xlsx', '*.xls']
        files = []
        for p in patterns:
            files.extend(sorted(glob(os.path.join(self.directory_path, p))))

        for file in files:
            try:
                df = self._get_dataframe(file)
            except Exception:
                # skip files that cannot be read
                continue

            # use filename (without extension) as id
            file_id = os.path.splitext(os.path.basename(file))[0]
            df[self.id_column] = str(file_id)

            # optionally filter ignored ids
            if hasattr(self, 'ignore_list') and file_id in getattr(self, 'ignore_list'):
                continue

            # ensure internal column names exist for unmapped items
            for col_name, mapped_name in self.data_mapping.items():
                if mapped_name is None and col_name not in df.columns:
                    df[col_name] = np.nan

            # compute iob/cob columns (functions expect numeric arrays)
            try:
                df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
            except Exception:
                df['iob'] = np.nan
            try:
                df['cob'] = calculate_total_cob(df['carbInput'].values, carb_absorption=0.8, ts_min=5, t_action_max_min=240)
            except Exception:
                df['cob'] = np.nan

            # store: split per-participant (first 80% -> train, remaining 20% -> test)
            # derive a patient key robustly from the filename/id
            raw_key = str(file_id)
            if "_" in raw_key:
                patient_key = raw_key.split("_")[1]
            else:
                patient_key = raw_key

            # sort by timestamp if present (internal name is '5minute_intervals_timestamp'),
            # or fallback to 'Timestamp' if reverse mapping didn't apply
            if '5minute_intervals_timestamp' in df.columns:
                df_sorted = df.sort_values(by='5minute_intervals_timestamp')
            elif 'Timestamp' in df.columns:
                df_sorted = df.sort_values(by='Timestamp')
            else:
                df_sorted = df

            n = len(df_sorted)
            split_idx = int(n * 0.8)
            # ensure at least one sample in train if there are any samples
            if split_idx == 0 and n > 0:
                split_idx = 1

            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()

            # store full dataframe and the splits
            self.all_dataframes[patient_key] = df_sorted.copy()
            self.train_dataframes[patient_key] = train_df
            self.test_dataframes[patient_key] = test_df

        #unique_patients = df[self.id_column].unique()
        #n_patients = len(unique_patients)
        #sorted_patients = sorted(unique_patients)
        #train_patients = set(sorted_patients[:int(0.9 * n_patients)])
        
        #for patient_id, group in df.groupby(self.id_column):
        #    if patient_id in train_patients:
        #        self.train_dataframes[str(patient_id)] = group
        #    else:
        #        self.test_dataframes[str(patient_id)] = group
        #    self.all_dataframes[str(patient_id)] = group
        #    print(patient_id)
    def _get_dataframe(self, file, calculate_iob = True):
        # read CSV or Excel file and return a DataFrame with internal column names applied
        _, ext = os.path.splitext(file)
        if ext.lower() == '.csv':
            df = pd.read_csv(file)
        else:
            # read excel (first sheet)
            df = pd.read_excel(file)

        # rename columns from source names to internal names if mapping provided
        if hasattr(self, 'reverse_data_mapping') and self.reverse_data_mapping:
            df = df.rename(columns=self.reverse_data_mapping)

        # convert cbg from mmol/L to mg/dL when present
        if 'cbg' in df.columns:
            # ensure numeric then convert (1 mmol/L = 18.01559 mg/dL)
            df['cbg'] = pd.to_numeric(df['cbg'], errors='coerce') * 18.01559

        return df

    def _get_dataset_specific_metadata(self):
        metadata = {
            '0001': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 28,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 24.49,
                'years_of_diabetes': 4
            },
            '0002': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 25,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 15.97,
                'years_of_diabetes': 3
            },
            '0003': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 20,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 24.38,
                'years_of_diabetes': 11
            },
            '0004': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 25,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 18.42,
                'years_of_diabetes': 2
            },
            '0005': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 46,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 27.72,
                'years_of_diabetes': 15
            },
            '0006': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 66,
                'age_range_low': 60,
                'age_range_high': 80,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 22.27,
                'years_of_diabetes': 38
            },
            '0007': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 51,
                'age_range_low': 50,
                'age_range_high': 70,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 26.15,
                'years_of_diabetes': 9
            },
            '0008': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': None,
                'age_range_low': None,
                'age_range_high': None,
                'biological_sex': None,
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': 21.88,
                'years_of_diabetes': None
            },
            '0009': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 75,
                'age_range_low': 70,
                'age_range_high': 90,
                'biological_sex': None,
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            },
            '0010': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 75,
                'age_range_low': 70,
                'age_range_high': 90,
                'biological_sex': None,
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            },
            '0011': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 68,
                'age_range_low': 60,
                'age_range_high': 80,
                'biological_sex': None,
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            },
            '0012': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': None,
                'age_range_low': None,
                'age_range_high': None,
                'biological_sex': None,
                'diagnosis_type': None,
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            },
            '0013': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': 70,
                'age_range_low': 70,
                'age_range_high': 90,
                'biological_sex': None,
                'diagnosis_type': 'type1',
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            },
            '0014': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': None,
                'age_range_low': None,
                'age_range_high': None,
                'biological_sex': None,
                'diagnosis_type': None,
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            },
            '0015': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age': None,
                'age_range_low': None,
                'age_range_high': None,
                'biological_sex': None,
                'diagnosis_type': None,
                'cgm_type': None,
                'device_type': 'open_loop',
                'device_name': None,
                'sensor_band': None,
                'sampling_rate': None,
                'bmi': None,
                'years_of_diabetes': None
            }
        }
        self.train_metadata = metadata
        self.test_metadata = metadata