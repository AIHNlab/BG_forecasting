import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob
import json

class DataloaderShanghai(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        if "Shanghai_T1DM" in self.directory_path:
            self.train_test_divide = 1011
        else:
            self.train_test_divide = 2090
        #self.train_test_divide_T2DM = 2090 #participants with a number higher than this will be put in the test set
        #self.train_test_divide_T1DM = 1011 #participants with a number higher than this will be put in the test set

    def load_data(self):
        #train_path = os.path.join(self.directory_path, "Shanghai")
        files = glob(self.directory_path + os.sep + "*.xls*")

        for file in files:
            if os.path.basename(file).startswith('~$'):
                continue  # skip temp files
            id = int(file.split(os.sep)[-1].split('_')[0])
            df, patient_id = self._get_dataframe(file)
            if id < self.train_test_divide:
                self.train_dataframes[patient_id] = df
            else:
                self.test_dataframes[patient_id] = df
            self.all_dataframes[patient_id] = df


        #return self.all_dataframes, self.train_dataframes, self.test_dataframes
    
    def _get_dataset_specific_metadata(self):
        """Extract patient metadata from an Excel file into a dictionary."""
        if "Shanghai_T1DM" in self.directory_path:
            filename = os.path.join(os.path.dirname(self.directory_path), 'Shanghai_T1DM_Summary.xlsx')
            df = pd.read_excel(filename, sheet_name='T1DM')
        else:
            filename = os.path.join(os.path.dirname(self.directory_path), 'Shanghai_T2DM_Summary.xlsx')
            df = pd.read_excel(filename, sheet_name='T2DM')
        
        data_mapping = {"Gender (Female=1, Male=2)": "biological_sex", "Age (years)": "age", "BMI (kg/m2)": "bmi", "Type of Diabetes": "diagonosis_type"}
        df = df.rename(columns=data_mapping)
        # Rename columns according to mapping
        og_metadata = df.set_index("Patient Number").to_dict(orient="index")
        # Map biological_sex values
        for patient, meta in og_metadata.items():
            meta["insulin_treatment"] = None
            if "biological_sex" in meta:
                if meta["biological_sex"] == 1:
                    meta["biological_sex"] = "female"
                elif meta["biological_sex"] == 2:
                    meta["biological_sex"] = "male"
            if "diagonosis_type" in meta:
                if meta["diagonosis_type"] == "T2DM":
                    meta["diagonosis_type"] = "type2"
                elif meta["diagonosis_type"] == "T1DM":
                    meta["diagonosis_type"] = "type1"

        #self.train_metadata = {}
        #self.test_metadata = {}

        for patient, meta in og_metadata.items():
            
            try:
                patient_num = int(patient.split(os.sep)[-1].split('_')[0])
            except Exception:
                continue
            if patient_num < self.train_test_divide:
                
                self.train_metadata[patient] = meta
            else:
                self.test_metadata[patient] = meta


    def _get_dataframe(self, file, calculate_iob=True):
        # Load the data from the Excel file
        df = pd.read_excel(file)
        # Rename relevant columns
        # Find and rename columns containing specific keywords
        for col in df.columns:
            if 'bolus insulin' in col.lower():
                df = df.rename(columns={col: 'bolus'})
                df['bolus'] = df['bolus'].apply(lambda x: np.nan if isinstance(x, str) else x)
            elif 'basal insulin' in col.lower():
                df = df.rename(columns={col: 'basal'})
                df['basal'] = df['basal'].apply(lambda x: np.nan if isinstance(x, str) else x)
            elif 'cgm' in col.lower():
                df = df.rename(columns={col: 'cbg'})
            elif 'date' in col.lower():
                df = df.rename(columns={col: '5minute_intervals_timestamp'})

        # Add columns that may not exist in the file
        df['carbInput'] = np.nan
        df['hr'] = np.nan
        if calculate_iob:
            try:
                df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
            except Exception as e:
                print(f"Error calculating IOB for {file}: {e}")
                df['iob'] = np.nan
            df['cob'] = np.nan  # or calculate_total_cob(...) if you want
        patient_id = file.split(os.sep)[-1].split('.')[0]
        return df, patient_id



