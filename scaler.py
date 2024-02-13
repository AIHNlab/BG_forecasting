import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self, data_handler, features):
        self.data_handler = data_handler
        self.features = features
        self.scaler = StandardScaler()

    def get_scaler(self):
        return self.scaler

    def fit_and_save(self, filename):
        self.scaler.fit(pd.concat(self.data_handler.get_train_dataframes().values())[self.features])
        # Save scaler for later use
        joblib.dump(self.scaler, "scalers" + os.sep + filename)

    def load_scaler(filename):
        return joblib.load("scalers" + os.sep + filename)