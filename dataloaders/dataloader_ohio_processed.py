import os
import pandas as pd

class Dataloader:
    def __init__(self, directory_path):
        self.directory_path = directory_path

        self.all_dataframes = {}
        self.train_dataframes = {}
        self.test_dataframes = {}

    def load_data(self):
        train_files = os.listdir(os.path.join(self.directory_path,"train"))
        for file in train_files:
            if file.endswith(".csv"):
                file_name = os.path.splitext(file)[0]
                self.train_dataframes[file_name] = pd.read_csv(os.path.join(self.directory_path,"train",file))
                self.all_dataframes[file_name] = pd.read_csv(os.path.join(self.directory_path,"train",file))
        
        test_files = os.listdir(os.path.join(self.directory_path,"test"))
        for file in test_files:
            if file.endswith(".csv"):
                file_name = os.path.splitext(file)[0]
                self.test_dataframes[file_name] = pd.read_csv(os.path.join(self.directory_path,"test",file))
                self.all_dataframes[file_name] = pd.read_csv(os.path.join(self.directory_path,"test",file))

        return self.all_dataframes, self.train_dataframes, self.test_dataframes