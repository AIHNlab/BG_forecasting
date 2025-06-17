import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob
import json

class DataloaderGlucobench(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        self.train_test_divide = 1558 #participatns with a number higerh than this wll be put in the test set
        if "colas" in self.directory_path:
            self.id_column = "id"
            self.data_mapping =  {"5minute_intervals_timestamp": "time", "cbg": "gl", "basal": None, "bolus": None, "hr": None, "carbInput": None}
            self.reverse_data_mapping = {v: k for k, v in self.data_mapping.items() if v is not None}
            self.metadata_mapping = {"age": "age","biological_sex": "gender", "bmi": "BMI", "diagonosis_type": "T2DM", "insulin_treatment": None}
            self.reverse_metadata_mapping = {v: k for k, v in self.metadata_mapping.items() if v is not None}


    def load_data(self):
        df = pd.read_csv(self.directory_path + ".csv")
        df = df.rename(columns=self.reverse_data_mapping)
        for col_name, mapped_name in self.data_mapping.items():
            if mapped_name is None:
                df[col_name] = np.nan
        df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
        df['cob'] = calculate_total_cob(df['carbInput'].values, carb_absorption=0.8, ts_min=5, t_action_max_min=240)

        unique_patients = df[self.id_column].unique()
        n_patients = len(unique_patients)
        sorted_patients = sorted(unique_patients)
        train_patients = set(sorted_patients[:int(0.9 * n_patients)])
        
        for patient_id, group in df.groupby(self.id_column):
            if patient_id in train_patients:
                self.train_dataframes[str(patient_id)] = group
            else:
                self.test_dataframes[str(patient_id)] = group
            self.all_dataframes[str(patient_id)] = group
            print(patient_id)


    def _get_dataset_specific_metadata(self):
        df = pd.read_csv(self.directory_path + ".csv")
        df = df.rename(columns=self.reverse_metadata_mapping)
        unique_patients = df[self.id_column].unique()
        n_patients = len(unique_patients)
        test_size = int(0.1 * n_patients)
        sorted_patients = sorted(unique_patients)
        train_patients = set(sorted_patients[:int(0.9 * n_patients)])
        for col_name, mapped_name in self.metadata_mapping.items():
            if mapped_name is None:
                df[col_name] = None
        if "colas" in self.directory_path:
            df['diagonosis_type'] = df['diagonosis_type'].map({True: 'type2', False: "normal"})
            df['biological_sex'] = df['biological_sex'].map({1: 'female', 0: 'male', 2: 'unknown_biological_sex'})

        for patient_id, group in df.groupby(self.id_column):
            if patient_id in train_patients:
                self.train_metadata[patient_id] = group.to_dict(orient='records')[0]
                #print(self.train_metadata[patient_id])
            else:
                self.test_metadata[patient_id] = group.to_dict(orient='records')[0]
                #print(self.test_metadata[patient_id])
        #print(self.train_metadata)

    def _get_dataframe(self, file, calculate_iob = True):
        pass


