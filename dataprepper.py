import torch
import numpy as np
from scaler import Scaler
from tqdm import tqdm
import pandas as pd

class DataPrepper:
    def __init__(self, participants, dataframes, specific_participant=None, 
                 forecast_steps=24, scaler_class_x = None, scaler_class_y = None, sequence_length = 25, indices_per_day=288,
                 feature_list = ['cbg', 'basal', 'carbInput', 'bolus'], target_list = ['cbg'],
                 step = 1, allowed_missing_values_rate = [0.5,1.0,1.0,1.0], allowed_missing_values_rate_target = [0.0], fill_types=None, experiment_path=None, history_of_days=0):

        self.participants = participants

        self.dataframes = dataframes
        self.specific_participant = specific_participant
        self.forecast_steps = forecast_steps
        self.feature_list = feature_list
        self.target_list = target_list
        self.df = None
        self.features = None
        self.target = None
        self.features_seq = None
        self.target_seq = None
        self.sequence_length = sequence_length
        self.indices_per_day = indices_per_day
        self.step = step
        self.allowed_missing_values_rate = np.array(allowed_missing_values_rate)
        self.allowed_missing_values_rate_target = np.array(allowed_missing_values_rate_target)
        self.history_of_days = history_of_days
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
        self.scaler_x = Scaler(dataframes=dataframes, features=self.feature_list, scaler=scaler_class_x, missing_mask=None, is_input=True, file_path=experiment_path)
        self.scaler_y = Scaler(dataframes=dataframes, features=self.target_list, scaler=scaler_class_y, missing_mask=None, is_input=False, file_path=experiment_path)   
    def make_features_and_targetpair(self):
        participant_sequences = []
        participant_targets = []
        for participant in tqdm(self.participants, desc="Processing participants"):
            if participant == self.specific_participant or self.specific_participant is None:
                df_participant = self.dataframes[participant]
                #df_participant_missing_mask = self.missing_mask[participant]
                features, targets = self._select_features_and_target(df_participant)
                #features_missing_mask, targets_missing_mask = self._select_features_and_target(df_participant_missing_mask)
                #is_correct = self.test_missing_mask(df_participant, self.feature_list, features_missing_mask)

                features_missing_mask = features[self.feature_list].isna()
                targets_missing_mask = targets[self.target_list].isna()


                features = self._normalize(features, self.scaler_x)
                targets = self._normalize(targets, self.scaler_y)

                self.handle_missing_values(features, self.feature_list, self.fill_types)
                self.handle_missing_values(targets, self.target_list, np.full(len(self.target_list), -8))
                

                # Create new columns for each feature for the previous history_of_days days
                for feature in self.feature_list:
                    for day in range(1, self.history_of_days + 1):
                        new_column_name = f"{feature}_prevday{day}"
                        features[new_column_name] = features[feature].shift(day * self.indices_per_day, fill_value=-9)

                for feature in self.target_list:
                    for day in range(1, self.history_of_days + 1):
                        new_column_name = f"{feature}_prevday{day}"
                        targets[new_column_name] = targets[feature].shift(day * self.indices_per_day, fill_value=-9)

                features_seq, target_seq = self._create_sequences(features, features_missing_mask,targets, targets_missing_mask)
                        # Check for missing values
                        #start_idx = day * indices_per_day
                        #if features[feature].iloc[:start_idx].isna().all():
                        #    features[new_column_name] = -1
                        #elif features[feature].iloc[:start_idx].isna().any():
                        #    features[new_column_name] = np.nan
                
                #sequences, targets = self._create_sequences_old(features, features_missing_mask,targets, targets_missing_mask)
                #features_seq, target_seq = torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
                participant_sequences.append(features_seq)
                participant_targets.append(target_seq)

        self.features_seq = torch.cat(participant_sequences, dim=0)
        self.target_seq = torch.cat(participant_targets, dim=0)
        return self.features_seq, self.target_seq

    #def _get_dataframes(self):
    #    dfs = []
    #    for participant in self.participants:
    #        if participant == self.specific_participant or self.specific_participant is None:
    #            dfs.append(self.dataframes.get_dataframe(participant))
    #    return dfs

    def handle_missing_values(self, df, features, fill_types):
        # Initialize the missing_mask dictionary to record original missing values
        #missing_mask = {key: df[features].isna() for key, df in self.dataframes.items()}
        #missing_mask = {}
        #for key, df in self.dataframes.items():
        #    missing_mask[key] = self.dataframes[key][features].copy(deep=True).isna().copy(deep=True)
        #for key, df in self.dataframes.items():
        #    self.dataframes[key][features] =  self.dataframes[key][features].copy(deep=True).fillna(-5).copy(deep=True)
#
        #for key, df in self.dataframes.items():
        #    import pandas as pd
        #    pd.concat([self.dataframes[key], missing_mask[key]], axis=1).to_csv("yo5.csv")
        #for df_mask in missing_mask.values():
            #pd.concat([df, df_mask], axis=1).to_csv("yo4.csv")
            #df_mask.shape
            #continue
        for feature, fill_type in zip(features, fill_types):
            if isinstance(fill_type, str):
                if fill_type == 'mean':
                    df[feature].fillna(df[feature].mean(), inplace=True)
                else:
                    df[feature].interpolate(method=fill_type, inplace=True)
                    df[feature].fillna(method='ffill', inplace=True)
                    df[feature].fillna(method='bfill', inplace=True)
            else:
                #import pandas as pd
                df[feature].fillna(fill_type, inplace=True)
#
                    #    exit()
        df.dropna(subset=features, inplace=True)
        
        #return missing_mask

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
        #features_current_day = [input_data.columns.get_loc(col) for col in target_column.columns]
        input_data_missing_mask = input_data_missing_mask.to_numpy()
        target_data_missing_mask = target_data_missing_mask.to_numpy()
        input_data = input_data.to_numpy()
        target_data = target_data.to_numpy()
        
        for i in range(0, len(input_data) - self.sequence_length - self.forecast_steps, self.step):
            
            if self.history_of_days > 0:
                sequence = input_data[i:i + self.sequence_length].copy()
                #sequence[self.sequence_length:self.sequence_length + self.forecast_steps, features_current_day] = 0
                target = target_data[i:i + self.sequence_length].copy()
            else:
                sequence = input_data[i:i + self.sequence_length]
                target = target_data[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
            sequence_missing_mask = input_data_missing_mask[i:i + self.sequence_length]

            # Calculate the rate of missing values in the sequence
            missing_values_rate = sequence_missing_mask.mean(axis=0)
            
            # If the rate of missing values for any feature is higher than allowed, skip this sequence
            if any(missing_values_rate > self.allowed_missing_values_rate):
                continue

            #target = target_column[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
            target_missing = target_data_missing_mask[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]


            missing_values_rate = target_missing.mean(axis=0)
            # If any target is missing, skip this sequence
            if any(missing_values_rate > self.allowed_missing_values_rate_target):
                continue
            sequences.append(sequence)
            targets.append(target)
        #sequences = np.array(sequences)
        #targets = np.array(targets)
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

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

