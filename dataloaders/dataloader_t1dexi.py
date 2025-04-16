import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob
import json

class DataloaderT1DEXI(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        self.train_test_divide = 1558 #participatns with a number higerh than this wll be put in the test set


    def load_data(self):
        train_path = os.path.join(self.directory_path, "TIDEXI_processed")
        train_files = glob(train_path + os.sep + "*.csv")

        for file in train_files:
            id = int(file.split(os.sep)[-1].split('_')[0])
            df, patient_id = self._get_dataframe(file)
            if id < self.train_test_divide:
                self.train_dataframes[patient_id] = df
            else:
                self.test_dataframes[patient_id] = df
            self.all_dataframes[patient_id] = df


        #return self.all_dataframes, self.train_dataframes, self.test_dataframes
    
    def _get_dataset_specific_metadata(self):
        metadata_path = os.path.join(self.directory_path, "metadata_t1dexi.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)  

        for patient_id in metadata.keys():
            id = int(patient_id.split('_')[0])
            if id < self.train_test_divide:
                self.train_metadata[patient_id] = metadata[patient_id]
            else:
                self.test_metadata[patient_id] = metadata[patient_id]


    def _get_dataframe(self, file, calculate_iob = True):
        # Load the data from the CSV file
        df = pd.read_csv(file)
        df['time'] = df['timestamp']
        if calculate_iob == True:
            df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
            #df['iob'] = pd.Series(df['iob']).rolling(window=12, min_periods=1).mean().to_numpy()
            df['cob'] = calculate_total_cob(df['carbInput'].values, carb_absorption=0.8, ts_min=5, t_action_max_min=240)
            #df['cob'] = pd.Series(df['cob']).rolling(window=12, min_periods=1).mean().to_numpy()
        patient_id = file.split(os.sep)[-1].split('.')[0]
        return df, patient_id



