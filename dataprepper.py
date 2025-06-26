import torch
import numpy as np
from scaler import Scaler
from tqdm import tqdm
import pandas as pd
from custom_dataset import SequenceDataset

class DataPrepper:
    def __init__(self, participants, dataframes, specific_participant=None, 
                 forecast_steps=24, scaler_class_x = None, scaler_class_y = None, sequence_length = 25, patch_size=288,
                 feature_list = ['cbg', 'basal', 'carbInput', 'bolus'], target_list = ['cbg'],
                 step = 1, allowed_missing_values_rate = [0.5,1.0,1.0], allowed_missing_values_rate_target = [0.0,1.0,1.0], fill_types=None, experiment_path=None, history_of_days=0,
                 test_target="cbg", mask_prob=0.0, chance_of_smbg=0.0, chance_feature_missing=0.0, metadata=None, rolling_mean_window=None, mask_future_target_covariates=True, disabled_covariates=[0,0,0], context_limit=None, baseline=False):

        self.participants = participants

        self.dataframes = dataframes
        self.specific_participant = specific_participant
        self.forecast_steps = forecast_steps
        self.feature_list = feature_list
        self.target_list = target_list
        self.metadata = metadata
        self.df = None
        self.features = None
        self.target = None
        self.features_seq = None
        self.target_seq = None
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.step = step
        self.allowed_missing_values_rate = np.array(allowed_missing_values_rate)
        self.allowed_missing_values_rate_target = np.array(allowed_missing_values_rate_target)
        self.history_of_days = history_of_days
        self.column_dict = {}
        i = 0
        self.n_features = len(feature_list)
        for feature in feature_list:
            self.column_dict[feature] = i
            i += 1
        self.test_target_index = self.column_dict[test_target]
        if experiment_path is not None:
            self.experiment_path = experiment_path
        else:
            self.experiment_path = "scalers"
        if fill_types is None:
            self.fill_types = ['linear' for _ in range(len(self.dataframes))]
        else:
            self.fill_types = fill_types
        #self.missing_mask = self.handle_missing_values(self.feature_list)
        # Initialize Scaler
        # Replace 0s with np.nan in all dataframes for features and targets
        for participant, df in self.dataframes.items():
            for feature in self.feature_list:
                if feature in df.columns:
                    df[feature].replace(0, np.nan, inplace=True)
            for target in self.target_list:
                if target in df.columns:
                    df[target].replace(0, np.nan, inplace=True)
                    
        self.scaler_x = Scaler(dataframes=dataframes, features=self.feature_list, scaler=scaler_class_x, missing_mask=None, is_input=True, file_path=experiment_path)
        self.scaler_y = Scaler(dataframes=dataframes, features=[test_target], scaler=scaler_class_y, missing_mask=None, is_input=False, file_path=experiment_path)
        self.hypoglycemia_threshold = self.scaler_x.transform_single_value(70)
        self.hyperglycemia_threshold = self.scaler_x.transform_single_value(180)
        self.test_target = test_target
        self.mask_prob = mask_prob
        self.chance_of_smbg = chance_of_smbg
        self.chance_feature_missing = chance_feature_missing
        self.rolling_mean_window = rolling_mean_window
        self.mask_future_target_covariates = mask_future_target_covariates
        self.disabled_covariates = disabled_covariates
        self.context_limit = context_limit
        self.baseline = baseline

    #@profile
    def make_features_and_targetpair(self):
        participant_datasets = []

        for participant in tqdm(self.participants, desc="Processing participants"):
            if participant == self.specific_participant or self.specific_participant is None:
                df_participant = self.dataframes[participant]#.rolling(window=12, min_periods=1).mean()
                #df_participant["cbg"] = df_participant["finger"]
                  # Add or remove columns as needed
                for i, feature in enumerate(self.feature_list):
                    if feature in df_participant.columns:
                        if self.rolling_mean_window is not None:
                            df_participant[feature] = df_participant[feature].rolling(window=self.rolling_mean_window[i], min_periods=1).mean()
                #df_participant = self.dataframes[participant]
                # Convert DataFrames to NumPy arrays
                features, targets = self._select_features_and_target(df_participant)

                features_missing_mask = features[self.feature_list].isna().values
                targets_missing_mask = targets[self.target_list].isna().values

                features = self._normalize(features, self.scaler_x).values
                targets = self._normalize(targets, self.scaler_x).values
                # Resize the features array to accommodate padding
                padded_features = np.zeros((features.shape[0] + self.sequence_length, features.shape[1]))

                for i, fill_value in enumerate(self.fill_types):
                    if isinstance(fill_value, (int, float)):  # Ensure fill_value is numeric
                        padded_features[:, i] = np.pad(
                            features[:, i], 
                            (self.sequence_length, 0), 
                            'constant', 
                            constant_values=fill_value
                        )
                    else:
                        raise ValueError(f"Fill type for feature {self.feature_list[i]} must be numeric.")

                features = padded_features  # Replace the original features with the padded version
                targets = np.pad(targets, ((self.sequence_length, 0), (0, 0)), 'constant', constant_values=-8)
                self.handle_missing_values(features, self.feature_list, self.fill_types)
                self.handle_missing_values(targets, self.target_list, np.full(len(self.target_list), -8))

                dataset = SequenceDataset(
                    input_data=features,
                    input_data_missing_mask=features_missing_mask,
                    target_data=targets,
                    target_data_missing_mask=targets_missing_mask,
                    sequence_length=self.sequence_length,
                    forecast_steps=self.forecast_steps,
                    step=self.step,
                    allowed_missing_values_rate=self.allowed_missing_values_rate,
                    allowed_missing_values_rate_target=self.allowed_missing_values_rate_target,
                    feature_list=self.feature_list,
                    target_list=self.target_list,
                    test_target=self.test_target,
                    patch_size=self.patch_size,
                    mask_prob=self.mask_prob,
                    chance_of_smbg=self.chance_of_smbg,
                    chance_feature_missing=self.chance_feature_missing,
                    metadata=self.metadata[participant],
                    mask_future_target_covariates=self.mask_future_target_covariates,
                    disabled_covariates=self.disabled_covariates,
                    context_limit=self.context_limit,
                    baseline=self.baseline
                    #hypo_threshold= self.hypoglycemia_threshold,
                    #hyper_threshold=self.hyperglycemia_threshold,
                )
                participant_datasets.append(dataset)

        # Combine all datasets
        full_dataset = torch.utils.data.ConcatDataset(participant_datasets)
        return full_dataset


    def handle_missing_values(self, df, features, fill_types):
        if isinstance(df, pd.DataFrame):
            for feature, fill_type in zip(features, fill_types):
                if isinstance(fill_type, str):
                    if fill_type == 'mean':
                        df[feature].fillna(df[feature].mean(), inplace=True)#.rolling(window=12, min_periods=1).mean()
                    else:
                        df[feature].interpolate(method=fill_type, inplace=True)
                        df[feature].fillna(method='ffill', inplace=True)
                        df[feature].fillna(method='bfill', inplace=True)
                else:
                    df[feature].fillna(fill_type, inplace=True)
            df.dropna(subset=features, inplace=True)
        else:  # If df is a NumPy array
            for i, feature in enumerate(features):
                if isinstance(fill_types[i], str):
                    if fill_types[i] == 'mean':
                        mean_value = np.nanmean(df[:, i])  # Compute mean ignoring NaN
                        df[:, i] = np.where(np.isnan(df[:, i]), mean_value, df[:, i])
                    else:
                        raise ValueError(f"Interpolation is not supported for NumPy arrays.")
                else:
                    df[:, i] = np.where(np.isnan(df[:, i]), fill_types[i], df[:, i])


    def _select_features_and_target(self, df):
        features = df[self.feature_list]
        target = df[self.target_list]
        return features, target

    def _normalize(self, data, scaler):
        if data.empty:
            return data
        scaler = scaler.get_scaler()
        if scaler is None:
            return data
        else:
            # Transform the data using the scaler
            transformed_data = scaler.transform(data)
            # Convert the NumPy array back to a DataFrame
            normalized_data = pd.DataFrame(transformed_data, index=data.index, columns=data.columns)
            return normalized_data

    #@profile
    def _create_sequences(self, input_data, input_data_missing_mask, target_data, target_data_missing_mask):
        sequences = []
        targets = []

        # Loop over sequences
        for i in range(0, len(input_data) - self.sequence_length - self.forecast_steps, self.step):
            sequence = input_data[i:i + self.sequence_length]
            target = target_data[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]

            sequence_missing_mask = input_data_missing_mask[i:i + self.sequence_length]
            target_missing = target_data_missing_mask[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]

            # Skip sequences with too many missing values
            if np.any(sequence_missing_mask.mean(axis=0) > self.allowed_missing_values_rate):
                continue
            if np.any(target_missing.mean(axis=0) > self.allowed_missing_values_rate_target):
                continue

            sequences.append(sequence)
            targets.append(target)

        # Convert to NumPy array once at the end
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        # Convert once to PyTorch tensors
        return torch.from_numpy(sequences), torch.from_numpy(targets)


    def _create_sequences_old(self, input_data, input_data_missing_mask, target_column, target_column_missing_mask):
        sequences = []
        targets = []
        features_current_day = [input_data.columns.get_loc(col) for col in target_column.columns]
        input_data_missing_mask = input_data_missing_mask.to_numpy()
        target_column_missing_mask = target_column_missing_mask.to_numpy()
        input_data = input_data.to_numpy()
        target_column = target_column.to_numpy()
        
        for i in range(0, len(input_data) - self.sequence_length - self.forecast_steps, self.step):
            
            if self.history_of_days > 0:
                sequence = input_data[i:i + self.sequence_length + self.forecast_steps].copy()
                sequence[self.sequence_length:self.sequence_length + self.forecast_steps, features_current_day] = 0
                #sequence[:, features_current_day][i+self.sequence_length:i+self.sequence_length+self.forecast_steps] = -8
            else:
                sequence = input_data[i:i + self.sequence_length]
            sequence_missing_mask = input_data_missing_mask[i:i + self.sequence_length]

            # Calculate the rate of missing values in the sequence
            missing_values_rate = sequence_missing_mask.mean(axis=0)
            
            # If the rate of missing values for any feature is higher than allowed, skip this sequence
            if any(missing_values_rate > self.allowed_missing_values_rate):
                continue

            target = target_column[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
            target_missing = target_column_missing_mask[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
            #target = input_data[i:i + self.sequence_length]
            #target_missing = input_data_missing_mask[i:i + self.sequence_length]
            #target = input_data[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
            #target_missing = input_data_missing_mask[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]

            missing_values_rate = target_missing.mean(axis=0)
            # If any target is missing, skip this sequence
            if any(missing_values_rate > self.allowed_missing_values_rate_target):
                continue
            #if np.any(target < 0):
            #    count_negative_values = np.sum(target < 0)
            #    print(f"Number of values less than 0: {count_negative_values}")
            sequences.append(sequence)
            targets.append(target)
        return sequences, targets
        # return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32

