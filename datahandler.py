import os
import pandas as pd
import pickle
from utils import save_dataframes, load_dataframes
import simplejson as json
from dataloaders.dataloader import Dataloader
from dataloaders.dataloader_ohio import DataloaderOhio
from dataloaders.dataloader_tidepool_sap100 import DataloaderTidepoolSAP100
from dataloaders.dataloader_tidepool_hcl150 import DataloaderTidepoolHCL150
from dataloaders.dataloader_t1dexi import DataloaderT1DEXI
from dataloaders.dataloader_fitbit_isphyncs import DataloaderFTBiSPHYNCS
from dataloaders.dataloader_merged import DataloaderMerged
from dataloaders.dataloader_glucobench import DataloaderGlucobench


class DataHandler:
    def __init__(self, dataloader_type, dataset_path, dataset_name):
        #self.dataloader = dataloader_type
        self.dataset_name = dataset_name
        self.dataloader = self.create_dataloader_instance(dataloader_type, dataset_path)
        
    def save_as_pickle(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + os.sep + 'all_dataframes.pkl', 'wb') as f:
            pickle.dump(self.dataloader.all_dataframes, f)
        with open(path + os.sep + 'train_dataframes.pkl', 'wb') as f:
            pickle.dump(self.dataloader.train_dataframes, f)
        with open(path + os.sep + 'test_dataframes.pkl', 'wb') as f:
            pickle.dump(self.dataloader.test_dataframes, f)
        with open(path + os.sep + 'train_metadata.json', 'w') as f:
            json.dump(self.dataloader.train_metadata, f, cls=CompactArrayEncoder, indent=4, ignore_nan=True)
        with open(path + os.sep + 'test_metadata.json', 'w') as f:
            json.dump(self.dataloader.test_metadata, f, cls=CompactArrayEncoder, indent=4, ignore_nan=True)

    def save_as_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        #save_dataframes(self.all_dataframes, path + os.sep + 'all')
        save_dataframes(self.dataloader.train_dataframes, path + os.sep + 'train')
        save_dataframes(self.dataloader.test_dataframes, path + os.sep + 'test')

    def load_data(self, load_from_pkl = True, save_as_pkl=True, save_as_csv=False, reload_metadata=False):
        standardized_datasets_path = os.path.join('standardized_datasets', self.dataset_name)
        if (os.path.exists(standardized_datasets_path) and load_from_pkl):
            if os.path.exists(standardized_datasets_path + os.sep + 'all_dataframes.pkl'):
                self.dataloader.all_dataframes = pd.read_pickle(standardized_datasets_path + os.sep + 'all_dataframes.pkl')
            if os.path.exists(standardized_datasets_path + os.sep + 'train_dataframes.pkl'):
                self.dataloader.train_dataframes = pd.read_pickle(standardized_datasets_path + os.sep + 'train_dataframes.pkl')
            if os.path.exists(standardized_datasets_path + os.sep + 'test_dataframes.pkl'):
                self.dataloader.test_dataframes = pd.read_pickle(standardized_datasets_path + os.sep + 'test_dataframes.pkl')
            if reload_metadata or not os.path.exists(standardized_datasets_path + os.sep + 'train_metadata.json'):
                self.dataloader.get_metadata()
            else:
                with open(standardized_datasets_path + os.sep + 'train_metadata.json', 'r') as f:
                    self.dataloader.train_metadata = json.load(f)
            if reload_metadata or not os.path.exists(standardized_datasets_path + os.sep + 'test_metadata.json'):
                self.dataloader.get_metadata()
            else:
                with open(standardized_datasets_path + os.sep + 'test_metadata.json', 'r') as f:
                    self.dataloader.test_metadata = json.load(f)
        else:
            self.dataloader.load_data()
            self.dataloader.get_metadata()
            if save_as_pkl:
                self.save_as_pickle(standardized_datasets_path)
        if save_as_csv:
            self.save_as_csv(standardized_datasets_path)
        if reload_metadata:
            with open(standardized_datasets_path + os.sep + 'train_metadata.json', 'w') as f:
                json.dump(self.dataloader.train_metadata, f, cls=CompactArrayEncoder, indent=4, ignore_nan=True)
            with open(standardized_datasets_path + os.sep + 'test_metadata.json', 'w') as f:
                json.dump(self.dataloader.test_metadata, f, cls=CompactArrayEncoder, indent=4, ignore_nan=True)
        #self.train_metadata, self.test_metadata = self.dataloader.get_metadata()
        

    def get_test_dataframes(self):
        return self.dataloader.test_dataframes
    
    def get_train_dataframes(self):
        return self.dataloader.train_dataframes
    
    def get_all_dataframes(self):
        return self.dataloader.all_dataframes

    def get_train_metadata(self):
        return self.dataloader.train_metadata

    def get_test_metadata(self):
        return self.dataloader.test_metadata

    def get_combined_df(self):
        return pd.concat(self.dataloader.all_dataframes.values())

    def get_dataset_name(self):
        return self.dataset_name
        
    def create_dataloader_instance(self, type, dataset_path):
        try:
            # Convert the string to a class
            cls = globals()[type]

            # Check if the class is a subclass of SuperClass
            if issubclass(cls, Dataloader):
                return cls(dataset_path)
            else:
                raise ValueError(f"{type} is not a subclass of SuperClass")
        except KeyError:
            raise ValueError(f"No class named {type} found")

class CompactArrayEncoder(json.JSONEncoder):
    def encode(self, o):
        parts = []
        for item in o:
            if isinstance(item, list):
                parts.append('[' + ', '.join(map(str, item)) + ']')
            else:
                parts.append(super().encode(item))
        return '[' + ', '.join(parts) + ']'
if __name__ == "__main__":
    #dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio_XML" ; data_handler = DataHandler("DataloaderOhio", dataset_path, dataset_name='Ohio2018')
    #dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio2020_XML" ; data_handler = DataHandler("DataloaderOhio", dataset_path, dataset_name='Ohio2020')
    #dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\T1DEXI" ; data_handler = DataHandler("DataloaderT1DEXI", dataset_path, dataset_name='T1DEXI')
    #dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Tidepool Data" ; data_handler = DataHandler("DataloaderTidepoolSAP100", dataset_path, dataset_name='Tidepool_SAP100')
    #dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Tidepool Data" ; data_handler = DataHandler("DataloaderTidepoolHCL150", dataset_path, dataset_name='Tidepool_HCL150')
    dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Glucobench\colas" ; data_handler = DataHandler("DataloaderGlucobench", dataset_path, dataset_name='Glucobench_Colas')
    data_handler.load_data(load_from_pkl = False, save_as_pkl=True, save_as_csv=True, reload_metadata=True)