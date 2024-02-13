import torch

class DataPrepper:
    def __init__(self, participants, dataframes, specific_participant=None, 
                 forecast_steps=24, scaler = None, sequence_length = 25,
                 feature_list = ['cbg', 'basal', 'carbInput', 'bolus'], target_list = ['cbg']):
        self.participants = participants
        self.dataframes = dataframes
        self.specific_participant = specific_participant
        self.forecast_steps = forecast_steps
        self.feature_list = feature_list
        self.target_list = target_list
        self.df = None
        self.features = None
        self.target = None
        self.scaler = scaler
        self.features_seq = None
        self.target_seq = None
        self.sequence_length = sequence_length

    def make_features_and_targetpair(self):
        participant_sequences = []
        participant_targets = []

        for participant in self.participants:
            if participant == self.specific_participant or self.specific_participant is None:
                df_participant = self.dataframes[participant]
                features, target = self._select_features_and_target(df_participant)
                features = self._normalize_features(features)
                features_seq, target_seq = self._create_sequences(features, target['cbg'].values)
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


    def _select_features_and_target(self, df):
        features = df[self.feature_list]
        target = df[self.target_list]
        return features, target

    def _normalize_features(self, features):
        if self.scaler == None:
            return features
        else:
            return self.scaler.transform(features)

    def _create_sequences(self, input_data, target_column):
        sequences = []
        targets = []
        for i in range(len(input_data) - self.sequence_length - self.forecast_steps):
            sequences.append(input_data[i:i + self.sequence_length])
            targets.append(target_column[i + self.sequence_length + self.forecast_steps - 1])
        return torch.FloatTensor(sequences), torch.FloatTensor(targets).view(-1, 1)
