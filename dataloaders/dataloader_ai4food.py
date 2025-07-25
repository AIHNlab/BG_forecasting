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
        # Load the data from the Excel file
        df = pd.read_csv(file)
        # Rename relevant columns
        # Find and rename columns containing specific keywords
        df = df.rename(columns={"glucose_value_in_mg_dl": 'cbg', "timestamp": '5minute_intervals_timestamp'})

        # Add columns that may not exist in the file
        df['carbInput'] = np.nan
        df['hr'] = np.nan
        df['basal'] = np.nan
        df['bolus'] = np.nan
        if calculate_iob:
            df['iob'] = np.nan
            df['cob'] = np.nan
        patient_id = file.split(os.sep)[-1].split('.')[0].replace('_glucose_levels', '')
        return df, patient_id



