import os
import matplotlib.pyplot as plt
import pandas as pd
from dataprepper import DataPrepper
from datahandler import DataHandler
from dataloaders.dataloader_tidepool_sap100 import Dataloader
from datapreprocessor import DataPreProcessor
from scaler import Scaler
from evaluator import Evaluator
import warnings
from sklearn.preprocessing import StandardScaler
from utils import IdentityTransformer
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import numpy as np
import shutil


def metadata(df):
    desc_stats = df['cbg'].describe()
    # Statistical Features
    #mean_glucose = df['cbg'].mean()
    #variance_glucose = df['cbg'].var()
    #covariance_glucose = df['cgm'].cov(df['cbg'])  # Covariance with itself is the variance
    #coef_variation = df['cbg'].std() / mean_glucose
    #skewness_glucose = df['cbg'].skew()
    #kurtosis_glucose = df['cbg'].kurtosis()
    #maximum_glucose = df['cbg'].max()
    #minimum_glucose = df['cbg'].min()
    #slope_glucose = np.polyfit(range(len(df['cbg'])), df['cbg'], 1)[0]
    # Generate descriptive statistics
    # Statistical Features from describe()
    number_of_samples = int(desc_stats['count'].item())
    number_of_cgm_samples = len(df['cbg'])
    mean_glucose = float(desc_stats['mean'].item())
    std_dev_glucose = float(desc_stats['std'].item())
    min_glucose = float(desc_stats['min'].item())
    percentile_25 = float(desc_stats['25%'].item())
    median_glucose = float(desc_stats['50%'].item())
    percentile_75 = float(desc_stats['75%'].item())
    max_glucose = float(desc_stats['max'].item())

    # Clinically Relevant Features
    # HbA1c ... not calculated from CGM data
    mean_glucose = mean_glucose  # already calculated
    percent_time_in_range = float((df['cbg'].between(70, 180).mean()) * 100)
    percent_time_tight_range = float((df['cbg'].between(70, 140).mean()) * 100)
    percent_time_low = float((df['cbg'] < 70).mean() * 100)
    percent_time_very_low = float((df['cbg'] < 54).mean() * 100)
    percent_time_high = float((df['cbg'] > 180).mean() * 100)
    percent_time_very_high = float((df['cbg'] > 250).mean() * 100)
    number_hypo_events = int((df['cbg'] < 70).sum())

    # Demographic Features
    # These are not calculated from the CGM data and would require additional demographic information.

    # Store results in a dictionary
    results = {
        #'Mean': mean_glucose,
        #'Variance': variance_glucose,
        #'Covariance': covariance_glucose,
        #'Coefficient of Variation': coef_variation,
        #'Skewness': skewness_glucose,
        #'Kurtosis': kurtosis_glucose,
        #'Maximum': maximum_glucose,
        #'Minimum': minimum_glucose,
        #'Slope': slope_glucose,
        'number_of_samples': number_of_samples,
        'number_of_cgm_samples': number_of_cgm_samples,
        'Mean': mean_glucose,
        'Standard Deviation': std_dev_glucose,
        'Minimum': min_glucose,
        '25th Percentile': percentile_25,
        'Median': median_glucose,
        '75th Percentile': percentile_75,
        'Maximum': max_glucose,
        '% Time in Range': percent_time_in_range,
        '% Time in Tight Range': percent_time_tight_range,
        '% Time Low': percent_time_low,
        '% Time Very Low': percent_time_very_low,
        '% Time High': percent_time_high,
        '% Time Very High': percent_time_very_high,
        '# Hypo Events': number_hypo_events,
        'Features': list(df.columns)
    }
    return results

data_handler = DataHandler("DataloaderOhio", "", dataset_name="Ohio2018")
data_handler.load_data()

all_dfs = data_handler.get_all_dataframes()
concatenated_df = pd.concat(all_dfs.values(), ignore_index=True)

#data_handler = DataHandler("DataloaderOhio", "", dataset_name="Ohio2020")
#data_handler.load_data()
#
#all_dfs = data_handler.get_all_dataframes()
#concatenated_df2 = pd.concat(all_dfs.values(), ignore_index=True)
#
#concatenated_df = pd.concat([concatenated_df1, concatenated_df2], ignore_index=True)
total = len(concatenated_df)
num_zeros = (concatenated_df['cbg'] == 0).sum()
num_nans = concatenated_df['cbg'].isna().sum()
missingness_rate = (num_nans + num_zeros) / total
concatenated_df.dropna(subset=['cbg'], inplace=True)
metadata_results = metadata(concatenated_df)
print(metadata_results)

# compute mean days per participant (sampling rate = 5 minutes)
n_participants = len(all_dfs)
total_samples = metadata_results['number_of_cgm_samples']  # or len(concatenated_df)
sampling_minutes = 5
samples_per_day = 1440 / sampling_minutes  # 1440 minutes per day
mean_days_per_participant = (total_samples / n_participants) / samples_per_day

print("Participants:", n_participants)
print("Total samples:", total_samples)
print("Mean days per participant:", mean_days_per_participant)
print("Missingness rate:", missingness_rate)