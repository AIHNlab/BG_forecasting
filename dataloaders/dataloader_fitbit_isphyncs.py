import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader

class DataloaderFTBiSPHYNCS(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)

    def load_data(self):
        train_path = os.path.join(self.directory_path, "FTBiSPHYNCS/train/")
        train_files = glob(train_path + os.sep + "*.csv")
        for file in train_files:
            df, patient_id = self._get_dataframe(file)
            self.train_dataframes[patient_id] = df 
            self.all_dataframes[patient_id] = df
        
        test_path = os.path.join(self.directory_path, "FTBiSPHYNCS/test/")
        test_files = glob(test_path + os.sep + "*.csv")
        for file in test_files:
            df, patient_id = self._get_dataframe(file)
            self.test_dataframes[patient_id] = df 
            self.all_dataframes[patient_id] = df
    
    def _get_dataset_specific_metadata(self):
        
        # make a dummy metadata dictionary with a column called 'cbg' for consistency with other dataloaders
        metadata = {}
        template_dict = {'number_of_samples': -1,
                         'number_of_cgm_samples': -1,
                         'age_range_low': -1,
                         'age_range_high': -1,
                         'biological_sex': 'na',
                         'diagnosis_type': 'na',
                         'cgm_type': 'na',
                         'sensor_band': 'na',
                         'sampling_rate': -1}
        for key in self.all_dataframes.keys():
            metadata[key] = template_dict
        
        for patient_id in metadata.keys():
            self.train_metadata[patient_id ] = metadata[patient_id]
            self.test_metadata[patient_id] = metadata[patient_id]

    def _get_dataframe(self, file):

        patient_id = file.split(os.sep)[-1].split('.')[0]
        df = pd.read_csv(file)
        
        # make a copy of the 'time' column and name it '1minute_intervals_timestamp' for consistency with other dataloaders
        df['1minute_intervals_timestamp'] = df['time'] 
        
        # add a column 'cbg' with all negative ones for consistency with other dataloaders
        df['cbg'] = -1
        
        df.set_index('1minute_intervals_timestamp', inplace=True)
        return df, patient_id