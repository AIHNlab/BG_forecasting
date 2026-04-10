import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from dataprepper import DataPrepper
from architectures.lstms import MirshekarianLSTM
from trainers.trainer_basic import TrainerBasic
from datahandler import DataHandler

from datapreprocessor import DataPreProcessor
from scaler import Scaler
from evaluator import Evaluator
import warnings
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action='ignore', category=FutureWarning)

from dataloaders.dataloader_tidepool_sap100 import Dataloader
data_loader = Dataloader(r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio2020")
data_handler = DataHandler(data_loader, dataset_name='Ohio2018')
data_handler.load_data()
#      5minute_intervals_timestamp  missing_cbg    cbg  finger  basal    hr       gsr  carbInput  bolus
#0                    5.473729e+06          0.0  283.0     NaN   1.00  69.0  0.013080        NaN    NaN
#1                    5.473730e+06          0.0  282.0     NaN   1.00  70.0  0.008778        NaN    NaN
#2                    5.473731e+06          0.0  281.0     NaN   1.00  71.0  0.008620        NaN    NaN
#3                    5.473732e+06          0.0  277.0     NaN   1.00  72.0  0.006384        NaN    NaN
#4                    5.473733e+06          0.0  267.0     NaN   1.00  71.0  0.004690        NaN    NaN
#...                           ...          ...    ...     ...    ...   ...       ...        ...    ...
#2842                 5.476572e+06          0.0  149.0     NaN   0.98  80.0  0.000068        NaN    NaN
#2843                 5.476573e+06          0.0  148.0     NaN   0.98  82.0  0.000070        NaN    NaN
#2844                 5.476574e+06          0.0  151.0     NaN   0.98  83.0  0.000075        NaN    NaN
#2845                 5.476575e+06          0.0  149.0   139.0   0.98  74.0  0.000070       22.0    2.4
#2846                 5.476576e+06          0.0  144.0     NaN   0.98  74.0  0.000082        NaN    NaN
#
#[2847 rows x 9 columns]
dataframes = data_handler.get_test_dataframes()
for key, df in dataframes.items():
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.plot(df['missing_cbg'])
    plt.title(f'missing_cbg for {key}')
    plt.xlabel('Index')
    plt.ylabel('missing_cbg')

    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.plot(df['cbg'])
    plt.title(f'cbg for {key}')
    plt.xlabel('Index')
    plt.ylabel('cbg')

    plt.tight_layout()
    plt.show()

