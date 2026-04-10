import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.interpolate import griddata

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

# Example usage:
carb_series = pd.Series([30, np.nan, 0, 50, np.nan, 20, np.nan, 0, np.nan])
carb_absorption = 0.8
ts_min = 5
#t_action_max_min = 240
N = 6  # Moving average window size
#total_cob_series = calculate_total_cob(carb_series, carb_absorption, ts_min, t_action_max_min)
#smoothed_cob_series = apply_moving_average(total_cob_series, N)
#print("Total COB Series:", total_cob_series)
#print("Smoothed COB Series:", smoothed_cob_series)

perform_interpolation = True
remove_random_patches = True
plot_fake_variance = False
apply_thresholds_bool = False
participant = "train_24b7a7a140092b0d2b1c4754efdd06a832493da1a9af866b304d16113de7abeb"
dataframes = pd.read_pickle('standardized_datasets/Tidepool_SAP100_prev/train_dataframes.pkl')
num_months = 1

def remove_patches(matrix, num_patches=5, patch_max_length=5):
    nrows, ncols = matrix.shape
    for _ in range(num_patches):
        row_start = random.randint(0, nrows - 1)
        patch_length = random.randint(1, patch_max_length)
        col_start = random.randint(0, ncols - patch_length)
        matrix[row_start, col_start:col_start + patch_length] = np.nan
    if ncols >= 48:
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
    
    # Apply total IOB calculation to 'bolus' column
    total_iob_series = calculate_total_iob(filtered_dataframe['bolus'].values, ts_min, t_action_max_min=240)
    filtered_dataframe['bolus'] = apply_moving_average(total_iob_series, N)
    
    # Apply total COB calculation to 'carbInput' column
    total_cob_series = calculate_total_cob(filtered_dataframe['carbInput'].values, carb_absorption, ts_min, t_action_max_min=240)
    filtered_dataframe['carbInput'] = apply_moving_average(total_cob_series, N)
    
    columns_to_plot = ['cbg', 'carbInput', 'bolus']
    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15), sharex=True)
    for i, column in enumerate(columns_to_plot):
        grouped = filtered_dataframe.groupby('date')[column].apply(list)
        max_length = max(grouped.apply(len), default=0)
        matrix = grouped.apply(lambda x: x + [np.nan]*(max_length - len(x))).tolist()
        matrix_array = np.array(matrix)
        if column == 'cbg' and perform_interpolation:
            x, y = np.indices(matrix_array.shape)
            points = np.array((x.flatten(), y.flatten())).T
            values = matrix_array.flatten()
            matrix_array = griddata(points[~np.isnan(values)], values[~np.isnan(values)], points, method='cubic').reshape(matrix_array.shape)
        if remove_random_patches:
            remove_patches(matrix_array, num_patches=60, patch_max_length=40)
        if apply_thresholds_bool:
            def apply_thresholds(value):
                if pd.isna(value):
                    return np.nan
                elif value <= 70:
                    return 0
                elif value <= 200:
                    return 1
                else:
                    return 2
            matrix_array = np.vectorize(apply_thresholds, otypes=[float])(matrix_array)
        time_labels = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]
        selected_time_labels = time_labels[::12]
        selected_time_positions = np.linspace(0, max_length-1, len(selected_time_labels))
        ax = axs[i]
        if plot_fake_variance:
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