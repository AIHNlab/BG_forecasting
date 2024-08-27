import os
import pandas as pd

def check_if_model_is_compatible_with_trainer(model, trainer):
    #This needs to be filled out everytime a new model or trainer is added to the system.
    supported_models_for_trainer_dict = {
        "TrainerBasic": ["MirshekarianLSTM"] 
    }
    if (type(trainer).__name__ == "Trainer" and type(model).__name__ in supported_models_for_trainer_dict["Trainer"]):
        print("Model is compatible with trainer.")
        return True
    else:
        print("Model is not compatible with trainer.")
        return False

def save_dataframes(dataframes, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Iterate over the dictionary
    for key, df in dataframes.items():
        # Create a filename based on the key
        filename = os.path.join(folder, f'{key}.csv')

        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)

def load_dataframes(folder):
    # Get a list of all CSV files in the folder
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    # Create a dictionary to hold the DataFrames
    dataframes = {}

    # Iterate over the files
    for file in files:
        # Create a key based on the filename
        key = os.path.splitext(file)[0]

        # Create a filename
        filename = os.path.join(folder, file)

        # Load the DataFrame from the CSV file
        df = pd.read_csv(filename)

        # Add the DataFrame to the dictionary
        dataframes[key] = df

    return dataframes

from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Returns self, nothing to compute here
        return self

    def partial_fit(self, X, y=None):
        # Since there's nothing to fit, just return self.
        # This maintains compatibility with incremental learning algorithms.
        return self

    def transform(self, X):
        # Returns the input data unchanged
        return X

    def inverse_transform(self, X):
        # Returns the input data unchanged
        return X