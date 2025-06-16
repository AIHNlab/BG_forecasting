import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import numpy as np
import random


def systematic_subsample_feature(sequence, feature_idx, sample_rate):
    """
    Systematically subsample a specific feature in the sequence and set non-sampled values to 0.
    
    Args:
        sequence (torch.Tensor): Input sequence of shape [sequence_length, n_features]
        feature_idx (int): Index of the feature to subsample
        sample_rate (float): Fraction of values to keep (between 0 and 1)
    
    Returns:
        torch.Tensor: Sequence with subsampled feature, other features unchanged
    """
    seq_length = sequence.shape[0]
    sample_size = int(seq_length * sample_rate)
    step = seq_length / sample_size
    start = random.uniform(0, step)
    
    # Create a copy of the sequence
    subsampled = sequence.clone()
    
    # Get indices to keep
    keep_indices = [int(start + step * i) % seq_length for i in range(sample_size)]
    
    # Create mask of zeros for the selected feature
    feature_mask = torch.zeros(seq_length)
    feature_mask[keep_indices] = 1
    
    # Apply mask only to the selected feature
    subsampled[:, feature_idx] *= feature_mask
    
    return subsampled

def random_subsample_feature(sequence, feature_idx, sample_rate):
    """
    Randomly subsample a specific feature in the sequence and set non-sampled values to 0.
    
    Args:
        sequence (torch.Tensor): Input sequence of shape [sequence_length, n_features]
        feature_idx (int): Index of the feature to subsample
        sample_rate (float): Fraction of values to keep (between 0 and 1)
    
    Returns:
        torch.Tensor: Sequence with randomly subsampled feature, other features unchanged
    """
    seq_length = sequence.shape[0]
    
    # Create a copy of the sequence
    subsampled = sequence.clone()
    
    # Create random mask for the selected feature
    feature_mask = torch.rand(seq_length) < sample_rate
    
    # Apply mask only to the selected feature
    subsampled[:, feature_idx] *= feature_mask
    
    return subsampled

class SequenceDataset(Dataset):
    def __init__(self, input_data, input_data_missing_mask, target_data, target_data_missing_mask, 
                 sequence_length, forecast_steps, step, allowed_missing_values_rate, allowed_missing_values_rate_target,
                 feature_list, target_list, test_target, patch_size, mask_prob, chance_of_smbg, chance_feature_missing, metadata,
                 mask_future_target_covariates=True, disabled_covariates=None, context_limit=None):#, mask_ratio, chance_of_feature_missing):
        
        self.input_data = input_data  # NumPy array
        self.input_data_missing_mask = input_data_missing_mask  # NumPy array
        self.target_data = target_data  # NumPy array
        self.target_data_missing_mask = target_data_missing_mask  # NumPy array
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.step = step
        self.allowed_missing_values_rate = allowed_missing_values_rate
        self.allowed_missing_values_rate_target = allowed_missing_values_rate_target

        self.feature_list = feature_list
        self.target_list = target_list
        self.test_target = test_target
        self.column_dict = {}
        i = 0
        self.n_features = len(feature_list)
        for feature in feature_list:
            self.column_dict[feature] = i
            i += 1
        self.test_target_index = self.column_dict[test_target]
        self.target_indices = [self.column_dict[feature] for feature in target_list]
        self.non_test_target_indices = [i for i in self.target_indices if i != self.test_target_index]

        self.valid_indices = self._find_valid_indices()
        self.patch_size = patch_size
        self.mask_prob = mask_prob
        self.chance_of_smbg = chance_of_smbg
        self.chance_feature_missing = chance_feature_missing
        required_metadata = ['diagnosis_type', 'biological_sex', 'device_type', 'age', 'bmi']
        self.mask_future_target_covariates = mask_future_target_covariates
        self.disabled_covariates = disabled_covariates
        self.context_limit = context_limit 

        # Define default values by field type
        default_values = {
            'diagnosis_type': '-', 'biological_sex': '-', 'device_type': '-', 'age': -1, 'bmi': -1                  # numeric field
        }

        # Create metadata dictionary with appropriate defaults for missing or None values
        self.metadata = {}
        for k in required_metadata:
            if k in metadata and metadata[k] is not None:
                self.metadata[k] = metadata[k]
            else:
                self.metadata[k] = default_values[k]
        

    def _find_valid_indices(self):
        """ Precompute valid indices to avoid checking conditions every time during __getitem__ """
        valid_indices = []
        for i in range(0, len(self.input_data) - self.sequence_length - self.forecast_steps, self.step):
            sequence_missing_mask = self.input_data_missing_mask[i:i + self.sequence_length]
            #target_missing = self.target_data_missing_mask[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
            target_missing = self.target_data_missing_mask[i:i + self.sequence_length]

            # Skip sequences with too many missing values
            if np.any(sequence_missing_mask.mean(axis=0) > self.allowed_missing_values_rate):
                continue
            if np.any(target_missing.mean(axis=0) > self.allowed_missing_values_rate_target):
                continue

            valid_indices.append(i)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # set target to sequence
        # mask sequence
        i = self.valid_indices[index]
        sequence = self.input_data[i:i + self.sequence_length]
        #target = self.target_data[i + self.sequence_length : i + self.sequence_length + self.forecast_steps]
        target = self.target_data[i:i + self.sequence_length]

        # Convert to PyTorch tensors
        sequence = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        if random.random() < self.chance_of_smbg:
            feature_to_subsample = self.test_target_index  # or any other feature index
            sequence = random_subsample_feature(sequence, feature_to_subsample, sample_rate=0.03)

        # Calculate number of complete patches
        n_patches = sequence.shape[0] // self.patch_size
        n_features = sequence.shape[1]
        
        # Generate random mask for all patches and features at once
        mask = torch.rand(n_patches, n_features) < self.mask_prob
        
        # Apply mask to each patch efficiently
        for p in range(n_patches):
            start_idx = p * self.patch_size
            end_idx = start_idx + self.patch_size
            # Use broadcasting to mask all selected features in the patch at once
            sequence[start_idx:end_idx, mask[p]] = -6
            #target[start_idx:end_idx, mask[p]] = -6

        # Randomly mask features with probability self.chance_feature_missing
        feature_mask = torch.rand(sequence.shape[1]) < self.chance_feature_missing
        if feature_mask.any():
            sequence[:, feature_mask] = -9
            target[:, feature_mask] = -9
        
        # Set disabled covariates to -9
        if not self.disabled_covariates:
            for i, disabled in enumerate(self.disabled_covariates):
                if disabled == 1 and i < sequence.shape[1]:
                    sequence[:, i] = -9
                    target[:, i] = -9
        
        # Set the non-cgm target values to -9 here to not predict covariates
        sequence[-self.forecast_steps:, :] = -9
        if self.mask_future_target_covariates:
            target[-self.forecast_steps:, self.non_test_target_indices] = -9
        sequence[-self.forecast_steps:, self.test_target_index] = -6

        if self.context_limit is not None and self.context_limit > 0:
            keep_steps = self.forecast_steps + self.context_limit
            if sequence.shape[0] > keep_steps:
                sequence[:-keep_steps, :] = -9
                target[:-keep_steps, :] = -9       

        # reshape to patch_size
        #sequence = self.periodicity_reshape(sequence, self.n_features, 'apply')
        #target = self.periodicity_reshape(target, self.n_features, 'apply')
        if self.metadata:
            return sequence, target, self.metadata
        else:
            return sequence, target

class EventBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, threshold_low=70, threshold_high=180):
        self.dataset = dataset
        self.batch_size = batch_size
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

        # Precompute class indices
        self.hypo_indices = []
        self.hyper_indices = []
        self.nonevent_indices = []

        MISSING_VALUES = {-9, -8}

        for idx in range(len(dataset)):
            _, target, *_ = dataset[idx]  # target shape: [seq_len, num_features]
            cgm = target[-24:, 0]  # assumes this index exists
            valid = ~torch.isin(cgm, torch.tensor(list(MISSING_VALUES)))
            cgm_valid = cgm[valid]

            if len(cgm_valid) == 0:
                self.nonevent_indices.append(idx)
                continue

            if (cgm_valid < self.threshold_low).float().mean() > 0.25:
                self.hypo_indices.append(idx)
            elif (cgm_valid > self.threshold_high).float().mean() > 0.25:
                self.hyper_indices.append(idx)
            else:
                self.nonevent_indices.append(idx)

        self.min_class_size = min(len(self.hypo_indices), len(self.hyper_indices), len(self.nonevent_indices))

    def __iter__(self):
        n_each = self.batch_size // 3
        total_batches = self.min_class_size // n_each

        for _ in range(total_batches):
            batch = random.sample(self.hypo_indices, n_each) + \
                    random.sample(self.hyper_indices, n_each) + \
                    random.sample(self.nonevent_indices, self.batch_size - 2 * n_each)
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.min_class_size // (self.batch_size // 3) * self.batch_size



#_______________________________________________
def remove_random_segment(sample, N):
    """
    Removes a random segment from N random channels of the sample.
    
    Args:
        sample (torch.Tensor): The input sample of shape [N_channels, N_timesteps].
        N (int): The number of random channels to modify.
        
    Returns:
        torch.Tensor: The modified sample with random segments removed from N random channels.
    """
    N_channels, N_timesteps = sample.shape
    modified_sample = sample.clone()
    
    # Randomly select N channels
    selected_channels = torch.randperm(N_channels)[:N]
    
    for channel in selected_channels:
        segment_length = torch.randint(1, N_timesteps + 1, (1,)).item()
        start_idx = torch.randint(0, N_timesteps - segment_length + 1, (1,)).item()
        modified_sample[channel, start_idx:start_idx + segment_length] = 0  # Set the segment to 0 or any other value indicating removal
    
    return modified_sample

class CustomDataset(Dataset):
    def __init__(self, data, labels, feature_list, target_list, history_of_days, forecast_steps, test_target, days_to_mask):
        self.data = data
        self.labels = labels
        self.column_dict = {}
        i = 0
        self.n_features = len(feature_list)
        for feature in feature_list:
            self.column_dict[feature] = i
            i += 1
            for day in range(1, history_of_days + 1):
                self.column_dict[f"{feature}_prevday{day}"] = i
                i += 1
        self.forecast_steps = forecast_steps
        self.test_target_index = self.column_dict[test_target]
        self.target_indices = [self.column_dict[feature] for feature in target_list]
        self.days_to_mask = days_to_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = remove_random_segment(sample.T, self.days_to_mask*self.n_features).T
        if self.forecast_steps > 1:
            #sample[-self.forecast_steps:, self.test_target_indices[0]] = 0
            for index in self.target_indices:
                sample[-self.forecast_steps:, index] = 0
        return sample, label