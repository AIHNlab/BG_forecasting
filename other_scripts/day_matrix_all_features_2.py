import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.interpolate import griddata

# Flags to control features
perform_interpolation = True
remove_random_patches = False  # New flag
plot_fake_variance = False
apply_thresholds_bool = False
#participant = "train_baf43b5cdb97c109223bbde6063c6632a4781a27d01f3c40024b2948bc1dc729"
participant = "train_24b7a7a140092b0d2b1c4754efdd06a832493da1a9af866b304d16113de7abeb"

dataframes = pd.read_pickle('standardized_datasets/Tidepool_SAP100/train_dataframes.pkl')

num_months = 1  # Example: 1 month


def remove_patches(matrix, num_patches=5, patch_max_length=5):
    """
    Removes random horizontal patches from the matrix, with variable patch length.
    :param matrix: The input matrix.
    :param num_patches: Number of patches to remove.
    :param patch_max_length: Maximum length of each horizontal patch.
    """
    nrows, ncols = matrix.shape
    for _ in range(num_patches):
        row_start = random.randint(0, nrows - 1)  # Select any row
        patch_length = random.randint(1, patch_max_length)  # Sample patch length between 1 and patch_max_length
        col_start = random.randint(0, ncols - patch_length)  # Ensure patch stays within bounds
        matrix[row_start, col_start:col_start + patch_length] = np.nan  # or use 0 for zeroing out
    if ncols >= 48:  # Check to ensure there are at least 24 columns
        matrix[-1, -48:] = np.nan

for name, dataframe in dataframes.items():
    if name != participant:
        if participant is not None:
            continue
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    dataframe['date'] = dataframe['time'].dt.date
    dataframe['year_month'] = dataframe['time'].dt.year * 12 + dataframe['time'].dt.month - 1

    available_periods = dataframe['year_month'].unique()
    possible_start_periods = [period for period in available_periods if period <= (max(available_periods) - num_months + 1)]
    if not possible_start_periods:
        print(f"No continuous {num_months}-month period available in the data for {name}.")
        continue
    start_period = possible_start_periods[3]
    
    end_period = start_period + num_months - 1
    filtered_dataframe = dataframe[(dataframe['year_month'] >= start_period) & (dataframe['year_month'] <= end_period)]
    
    columns_to_plot = ['cbg', 'carbInput', 'basal']
    
    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15), sharex=True)
    
    for i, column in enumerate(columns_to_plot):
        if column == 'basal':
            filtered_dataframe[column] = filtered_dataframe[column].rolling(window=12, min_periods=1).mean()
        grouped = filtered_dataframe.groupby('date')[column].apply(list)
        max_length = max(grouped.apply(len), default=0)
        
        matrix = grouped.apply(lambda x: x + [np.nan]*(max_length - len(x))).tolist()
        matrix_array = np.array(matrix)


        if column == 'cbg' and perform_interpolation:
            # Perform interpolation
            x, y = np.indices(matrix_array.shape)
            points = np.array((x.flatten(), y.flatten())).T
            values = matrix_array.flatten()
            matrix_array = griddata(points[~np.isnan(values)], values[~np.isnan(values)], points, method='cubic').reshape(matrix_array.shape)
        
        if remove_random_patches:
            remove_patches(matrix_array, num_patches=60, patch_max_length=40)  # Adjust `num_patches` and `patch_size` as needed
        
        if apply_thresholds_bool:
            def apply_thresholds(value):
                if pd.isna(value):  # Keep NaN values as they are
                    return np.nan
                elif value <= 70:
                    return 0  # Class 1 (low)
                elif value <= 200:
                    return 1  # Class 2 (medium)
                else:
                    return 2  # Class 3 (high)
                
            matrix_array = np.vectorize(apply_thresholds, otypes=[float])(matrix_array)

        time_labels = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]
        selected_time_labels = time_labels[::12]
        selected_time_positions = np.linspace(0, max_length-1, len(selected_time_labels))
        
        ax = axs[i]
        if plot_fake_variance == True:
            cax = ax.imshow(matrix_array/100, aspect='auto', cmap='plasma', interpolation='nearest')
        else:
            cax = ax.imshow(matrix_array, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f'{column.capitalize()}')
        if i == len(columns_to_plot) - 1:
            ax.set_xlabel('Time of Day')
        ax.set_ylabel('Day')
        ax.set_yticks(range(0, len(matrix_array), 7))
        ax.set_yticklabels([str(grouped.index[j]) for j in range(0, len(matrix_array), 7)])
        ax.set_xticks(selected_time_positions)
        ax.set_xticklabels(selected_time_labels, rotation=45)
        ax.set_xticks(np.arange(-0.5, max_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix_array), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
        ax.tick_params(which='minor', size=0)
        fig.colorbar(cax, ax=ax, orientation='vertical', label='Value', pad=0.01)
    
    start_year, start_month = divmod(start_period + 1, 12)
    end_year, end_month = divmod(end_period + 1, 12)
    plt.suptitle(f'Day Matrix for {name} ({start_month}/{start_year}-{end_month}/{end_year})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()