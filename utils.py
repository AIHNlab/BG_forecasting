import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
    
def calculate_insulin_availability_and_iob_single_delivery(insulin, ts_min, t_action_max_min):
    tmax = 55
    ke = 0.138
    result_array_size = t_action_max_min // ts_min
    q1 = np.zeros(result_array_size)
    q2 = np.zeros(result_array_size)
    I = np.zeros(result_array_size)
    q4 = np.zeros(result_array_size)
    dq1 = np.zeros(result_array_size)
    dq2 = np.zeros(result_array_size)
    dI = np.zeros(result_array_size)
    dq4 = np.zeros(result_array_size)
    for tt in range(result_array_size - 1):
        if tt == 0:
            dq1[tt] = -(q1[tt] / tmax) + (insulin / ts_min)
        else:
            dq1[tt] = -(q1[tt] / tmax)
        dq2[tt] = (q1[tt] / tmax) - (q2[tt] / tmax)
        dI[tt] = (q2[tt] / tmax) - ke * I[tt]
        dq4[tt] = ke * I[tt]
        q1[tt + 1] = q1[tt] + dq1[tt] * ts_min
        q2[tt + 1] = q2[tt] + dq2[tt] * ts_min
        I[tt + 1] = I[tt] + dI[tt] * ts_min
        q4[tt + 1] = q4[tt] + dq4[tt] * ts_min
    ins_availability = I
    iob = insulin - q4
    return ins_availability, iob

def calculate_total_iob(time_series, ts_min, t_action_max_min):
    time_series = np.nan_to_num(time_series)
    total_iob_series = np.zeros(len(time_series))
    for t in range(len(time_series)):
        if time_series[t] > 0:
            _, iob = calculate_insulin_availability_and_iob_single_delivery(time_series[t], ts_min, t_action_max_min)
            for offset in range(len(iob)):
                if t + offset < len(time_series):
                    total_iob_series[t + offset] += iob[offset]
    return total_iob_series

# COB calculation functions
def calculate_carb_availability_and_cob_single_meal(meal_carbs, carb_absorption, ts_min, t_action_max_min):
    tmax = 40
    result_array_size = t_action_max_min // ts_min
    q1 = np.zeros(result_array_size)
    q2 = np.zeros(result_array_size)
    q3 = np.zeros(result_array_size)
    dq1 = np.zeros(result_array_size)
    dq2 = np.zeros(result_array_size)
    dq3 = np.zeros(result_array_size)
    for tt in range(result_array_size - 1):
        if tt == 0:
            dq1[tt] = -(q1[tt] / tmax) + (carb_absorption * meal_carbs / ts_min)
        else:
            dq1[tt] = -(q1[tt] / tmax)
        dq2[tt] = (q1[tt] / tmax) - (q2[tt] / tmax)
        dq3[tt] = q2[tt] / tmax
        q1[tt + 1] = q1[tt] + dq1[tt] * ts_min
        q2[tt + 1] = q2[tt] + dq2[tt] * ts_min
        q3[tt + 1] = q3[tt] + dq3[tt] * ts_min
    meal_availability = q2
    cob = carb_absorption * meal_carbs - q3
    return meal_availability, cob

def calculate_total_cob(time_series, carb_absorption, ts_min, t_action_max_min):
    time_series = np.nan_to_num(time_series)
    total_cob_series = np.zeros(len(time_series))
    for t in range(len(time_series)):
        if time_series[t] > 0:
            _, cob = calculate_carb_availability_and_cob_single_meal(time_series[t], carb_absorption, ts_min, t_action_max_min)
            for offset in range(len(cob)):
                if t + offset < len(time_series):
                    total_cob_series[t + offset] += cob[offset]
    return total_cob_series

def apply_moving_average(series, window_size):
    return pd.Series(series).rolling(window=window_size, min_periods=1).mean().to_numpy()

def get_thresholded_events(predictions, hyper_thresh=180, hypo_thresh=70):
    # Build baseline alarm probabilities from forecasts:
    # For each sample i, look at the first 12 forecast steps (1 hour). If there is any run
    # of 3 consecutive predicted values within that window that are > hyper threshold -> alarm=1.
    # Similarly for hypo (< hypo threshold). Otherwise alarm=0.
    #try:
    #    hyper_thresh = prepper.hyperglycemia_threshold
    #    hypo_thresh = prepper.hypoglycemia_threshold
    #except Exception:
        # fallback to typical defaults if prepper doesn't expose thresholds
    # Ensure predictions shape is (N, H, C) and create per-sample binary probabilities
    preds = np.array(predictions)
    n_samples = preds.shape[0]
    hyper_probs = np.zeros(n_samples, dtype=float)
    hypo_probs = np.zeros(n_samples, dtype=float)

    # Window length in forecast steps corresponding to 1 hour (12 timesteps)
    horizon_window = min(12, preds.shape[1]) if preds.ndim >= 2 else 12

    for i in range(n_samples):
        # Extract first `horizon_window` forecasted values and flatten
        if preds.ndim == 3:
            window_vals = preds[i, :horizon_window, :].reshape(-1)
        elif preds.ndim == 2:
            window_vals = preds[i, :horizon_window].reshape(-1)
        else:
            # Unexpected shape - skip
            continue

        # Treat NaNs as non-events
        valid_mask = ~np.isnan(window_vals)
        if valid_mask.sum() == 0:
            continue

        hyper_mask = valid_mask & (window_vals > hyper_thresh)
        hypo_mask = valid_mask & (window_vals < hypo_thresh)

        # Check for any run of 3 consecutive True values
        if len(hyper_mask) >= 3:
            conv = np.convolve(hyper_mask.astype(int), np.ones(3, dtype=int), mode='valid')
            if np.any(conv >= 3):
                hyper_probs[i] = 1.0

        if len(hypo_mask) >= 3:
            conv = np.convolve(hypo_mask.astype(int), np.ones(3, dtype=int), mode='valid')
            if np.any(conv >= 3):
                hypo_probs[i] = 1.0
    
    return hyper_probs, hypo_probs