import os
import joblib
import pandas as pd
import numpy as np

class Scaler:
    def __init__(self, dataframes, features, scaler, file_path, is_input, missing_mask=None):
        self.dataframes = dataframes
        self.features = features
        self.scaler = scaler
        #scaler_filename = scaler.__class__.__name__ + "_" + data_handler.get_dataset_name()
        self.missing_mask = missing_mask
        self.file_path = file_path
        #self.history_of_days = history_of_days
        if is_input:
            scaler_filename = file_path + os.sep + "scaler_input.pkl"
        else:
            scaler_filename = file_path + os.sep + "scaler_target.pkl"
        if not os.path.exists(scaler_filename):
            self.fit_and_save(scaler_filename)
            print("fitting and saving scaler to file: ", scaler_filename)
        else:
            #self.load_scaler(scaler_filename)
            print("loading scaler from file: ", scaler_filename)
            self.load_scaler(scaler_filename)


    def get_scaler(self):
        return self.scaler

    def fit_and_save(self, scaler_filename):
        train_dataframes = self.dataframes
        #indices_per_day = 288
        train_dataframes = pd.concat(train_dataframes.values())[self.features]
        #for feature in self.features:
        #    for day in range(1, self.history_of_days + 1):
        #        new_column_name = f"{feature}_prevday{day}"
        #        train_dataframes[new_column_name] = train_dataframes[feature]
        if self.missing_mask is None:
            self.scaler.fit(train_dataframes)
        else:
            # Fit the scaler on the non-missing values of each DataFrame
            print(self.missing_mask.keys())
            for key in train_dataframes.keys():
                if key not in self.missing_mask:
                    not_missing_mask = np.ones_like(train_dataframes[key], dtype=bool)
                else:
                    not_missing_mask = ~self.missing_mask[key]
                self.scaler.partial_fit(train_dataframes[key][self.features][not_missing_mask])
        # Save scaler for later use
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        joblib.dump(self.scaler, scaler_filename)

    def load_scaler(self, scaler_filename):
        self.scaler = joblib.load(scaler_filename)
    
