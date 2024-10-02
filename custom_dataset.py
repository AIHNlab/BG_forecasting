import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

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