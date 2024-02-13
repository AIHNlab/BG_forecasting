import os
import pandas as pd
import pickle
from utils import save_dataframes, load_dataframes

class DataHandler:
    def __init__(self, dataloader, dataset_name):
        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.all_dataframes = {}
        self.train_dataframes = {}
        self.test_dataframes = {}
        
    def save_as_pickle(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + os.sep + 'all.pkl', 'wb') as f:
            pickle.dump(self.all_dataframes, f)
        with open(path + os.sep + 'train.pkl', 'wb') as f:
            pickle.dump(self.train_dataframes, f)
        with open(path + os.sep + 'test.pkl', 'wb') as f:
            pickle.dump(self.test_dataframes, f)

    def save_as_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        #save_dataframes(self.all_dataframes, path + os.sep + 'all')
        save_dataframes(self.train_dataframes, path + os.sep + 'train')
        save_dataframes(self.test_dataframes, path + os.sep + 'test')

    def load_data(self, load_from_pkl = True, save_as_pkl=True, save_as_csv=True):
        standardized_datasets_path = os.path.join('standardized_datasets', self.dataset_name)
        if os.path.exists(standardized_datasets_path):
            if os.path.exists(standardized_datasets_path + os.sep + 'all.pkl'):
                self.all_dataframes = pd.read_pickle(standardized_datasets_path + os.sep + 'all.pkl')
            if os.path.exists(standardized_datasets_path + os.sep + 'train.pkl'):
                self.train_dataframes = pd.read_pickle(standardized_datasets_path + os.sep + 'train.pkl')
            if os.path.exists(standardized_datasets_path + os.sep + 'test.pkl'):
                self.test_dataframes = pd.read_pickle(standardized_datasets_path + os.sep + 'test.pkl')
        else:
            self.all_dataframes, self.train_dataframes, self.test_dataframes = self.dataloader.load_data()
            if save_as_pkl:
                self.save_as_pickle(standardized_datasets_path)
        if save_as_csv:
            self.save_as_csv(standardized_datasets_path)


    def get_test_dataframes(self):
        return self.test_dataframes
    
    def get_train_dataframes(self):
        return self.train_dataframes
    
    def get_all_dataframes(self):
        return self.all_dataframes

    def get_combined_df(self):
        return pd.concat(self.all_dataframes.values())

