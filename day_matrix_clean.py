import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.interpolate import griddata

# Constants
TS_MIN = 5
T_ACTION_MAX_MIN_INSULIN = 240
T_ACTION_MAX_MIN_CARB = 240
MOVING_AVG_WINDOW_SIZE = 12
CARB_ABSORPTION = 0.8

# Insulin calculations
def calculate_insulin_iob(insulin, ts_min, t_action_max_min):
    tmax, ke = 55, 0.138
    steps = t_action_max_min // ts_min
    q1, q2, I, q4 = [np.zeros(steps) for _ in range(4)]

    for t in range(steps - 1):
        dq1 = -(q1[t] / tmax) + (insulin / ts_min if t == 0 else 0)
        dq2 = (q1[t] - q2[t]) / tmax
        dI = q2[t] / tmax - ke * I[t]
        dq4 = ke * I[t]

        q1[t+1], q2[t+1] = q1[t] + dq1 * ts_min, q2[t] + dq2 * ts_min
        I[t+1], q4[t+1] = I[t] + dI * ts_min, q4[t] + dq4 * ts_min

    return I, insulin - q4

def total_iob_series(insulin_series, ts_min, t_action_max_min):
    insulin_series = np.nan_to_num(insulin_series)
    total_iob = np.zeros_like(insulin_series)

    for t, dose in enumerate(insulin_series):
        if dose > 0:
            _, iob = calculate_insulin_iob(dose, ts_min, t_action_max_min)
            total_iob[t:t+len(iob)] += iob[:len(insulin_series)-t]

    return total_iob

# Carbohydrate calculations
def calculate_carb_cob(carbs, absorption, ts_min, t_action_max_min):
    tmax = 40
    steps = t_action_max_min // ts_min
    q1, q2, q3 = [np.zeros(steps) for _ in range(3)]

    for t in range(steps - 1):
        dq1 = -(q1[t] / tmax) + (absorption * carbs / ts_min if t == 0 else 0)
        dq2 = (q1[t] - q2[t]) / tmax
        dq3 = q2[t] / tmax

        q1[t+1], q2[t+1], q3[t+1] = q1[t] + dq1 * ts_min, q2[t] + dq2 * ts_min, q3[t] + dq3 * ts_min

    return q2, absorption * carbs - q3

def total_cob_series(carb_series, absorption, ts_min, t_action_max_min):
    carb_series = np.nan_to_num(carb_series)
    total_cob = np.zeros_like(carb_series)

    for t, meal in enumerate(carb_series):
        if meal > 0:
            _, cob = calculate_carb_cob(meal, absorption, ts_min, t_action_max_min)
            total_cob[t:t+len(cob)] += cob[:len(carb_series)-t]

    return total_cob

# Moving average smoothing
def smooth_series(series, window_size):
    return pd.Series(series).rolling(window_size, min_periods=1).mean().to_numpy()

# Random patches for missing data simulation
def remove_random_patches(matrix, num_patches=60, max_patch_length=40):
    rows, cols = matrix.shape
    for _ in range(num_patches):
        row, length = random.randint(0, rows - 1), random.randint(1, max_patch_length)
        start_col = random.randint(0, cols - length)
        matrix[row, start_col:start_col+length] = np.nan
    if cols >= 48:
        matrix[-1, -48:] = np.nan

def remove_interval_patches(matrix, n_intervals=12, patch_length=None, prob_remove=0.3):
    """
    Remove patches at specified intervals throughout the day.
    
    Args:
        matrix: numpy array of shape (days, timepoints)
        n_intervals: number of intervals to divide each day into
        patch_length: length of patches to remove (if None, uses interval width)
        prob_remove: probability of removing a patch at each interval
    """
    rows, cols = matrix.shape
    interval_width = cols // n_intervals
    
    # If patch_length not specified, use interval width
    if patch_length is None:
        patch_length = interval_width
    
    for row in range(rows):
        for interval in range(n_intervals):
            if random.random() < prob_remove:
                # Align start exactly with interval boundaries
                start = interval * interval_width
                # Make patch length match interval width
                end = start + patch_length
                matrix[row, start:end] = np.nan
    
    # Always remove last 48 points of last day
    if cols >= 48:
        matrix[-1, -48:] = np.nan
    
    return matrix

def simulate_smbg_measurements(matrix, n_intervals=12, measurements_per_interval=1, prob_measure=0.3):
    """
    Simulate SMBG measurements by keeping only a few values per interval.
    
    Args:
        matrix: numpy array of shape (days, timepoints)
        n_intervals: number of intervals to divide each day into
        measurements_per_interval: max number of measurements to keep per interval
        prob_measure: probability of taking a measurement in each interval
    
    Returns:
        numpy.ndarray: Matrix with only simulated SMBG measurements
    """
    rows, cols = matrix.shape
    interval_width = cols // n_intervals
    smbg_matrix = np.full_like(matrix, np.nan)
    
    for row in range(rows):
        for interval in range(n_intervals):
            if random.random() < prob_measure:
                start = interval * interval_width
                end = start + interval_width
                # Get valid values in this interval
                interval_data = matrix[row, start:end]
                valid_indices = np.where(~np.isnan(interval_data))[0]
                
                if len(valid_indices) > 0:
                    # Randomly select up to measurements_per_interval values
                    n_measurements = min(measurements_per_interval, len(valid_indices))
                    selected_indices = np.random.choice(
                        valid_indices, 
                        size=n_measurements, 
                        replace=False
                    )
                    # Keep only selected measurements
                    for idx in selected_indices:
                        smbg_matrix[row, start + idx] = interval_data[idx]
    
    return smbg_matrix

# Main processing and plotting logic
def process_and_plot(dataframes, participant, weeks=4, interpolate=True, 
                    remove_patches=True, use_intervals=True, n_intervals=12, 
                    apply_thresholds=False, simulate_smbg=False):
    for name, df in dataframes.items():
        if participant and name != participant:
            continue

        df['time'] = pd.to_datetime(df['time'])
        df['date'] = df['time'].dt.date
        # Calculate week numbers instead of months
        df['year_week'] = df['time'].dt.isocalendar().week + df['time'].dt.year * 52

        periods = sorted(df['year_week'].unique())
        valid_start_periods = [p for p in periods if p <= max(periods) - weeks + 1]
        if not valid_start_periods:
            print(f"No {weeks}-week continuous period available for {name}.")
            continue

        start_period = valid_start_periods[3]
        end_period = start_period + weeks - 1
        df = df[df['year_week'].between(start_period, end_period)]

        # Calculate and smooth IOB and COB
        df['bolus'] = smooth_series(total_iob_series(df['bolus'].values, TS_MIN, T_ACTION_MAX_MIN_INSULIN), MOVING_AVG_WINDOW_SIZE)
        df['carbInput'] = smooth_series(total_cob_series(df['carbInput'].values, CARB_ABSORPTION, TS_MIN, T_ACTION_MAX_MIN_CARB), MOVING_AVG_WINDOW_SIZE)
        # Define base columns and conditionally add optional columns
        columns = ['cbg', 'carbInput', 'bolus']
        
        # Conditionally add optional columns
        if 'hr' in df.columns:
            df['hr'] = smooth_series(df['hr'].values, MOVING_AVG_WINDOW_SIZE)
            columns.append('hr')
            
        if 'basal' in df.columns:
            df['basal'] = smooth_series(df['basal'].values, MOVING_AVG_WINDOW_SIZE)
            columns.append('basal')
        fig, axs = plt.subplots(len(columns), 1, figsize=(10, 15), sharex=True)

        for i, col in enumerate(columns):
            daily_matrix = df.groupby('date')[col].apply(list)
            max_len = daily_matrix.str.len().max()
            matrix = np.array([d + [np.nan]*(max_len - len(d)) for d in daily_matrix])

            # Add SMBG simulation for CGM column
            if col == 'cbg' and simulate_smbg:
                matrix = simulate_smbg_measurements(
                    matrix,
                    n_intervals=n_intervals,
                    measurements_per_interval=1,
                    prob_measure=0.3
                )
                df_smooth = pd.DataFrame(matrix)
                smoothed = df_smooth.rolling(window=MOVING_AVG_WINDOW_SIZE, 
                                          min_periods=1, 
                                          axis=1).mean().to_numpy()
                matrix = smoothed
            elif col in ['cbg', 'hr', 'basal'] and interpolate:
                x, y = np.indices(matrix.shape)
                valid = ~np.isnan(matrix)
                if valid.any():  # Only interpolate if there are valid values
                    matrix = griddata(
                        (x[valid], y[valid]), 
                        matrix[valid], 
                        (x, y), 
                        method='cubic'
                    )
        
            if remove_patches:
                if use_intervals:
                    matrix = remove_interval_patches(matrix, 
                                                  n_intervals=12,  # 2-hour intervals
                                                  patch_length=None, 
                                                  prob_remove=0.1)
                else:
                    remove_random_patches(matrix)

            img = axs[i].imshow(matrix, aspect='auto', cmap='viridis')
            axs[i].set_title(col.capitalize())
            axs[i].set_ylabel('Day')
            plt.colorbar(img, ax=axs[i], label='Value')
            
            #n_divisions = 12
            # Add vertical lines to divide days
            for div in range(1, n_intervals):
                axs[i].axvline(x=max_len * div / n_intervals, color='white', linestyle='-', linewidth=0.5, alpha=0.5)

            # Add grid for days - align with actual row boundaries
            axs[i].set_yticks(np.arange(matrix.shape[0]))
            axs[i].grid(True, which='major', axis='y', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Set major ticks for both axes
            time_points = np.linspace(0, max_len-1, n_intervals+1)
            axs[i].set_xticks(time_points)
            axs[i].set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)  # Add minor ticks for grid alignment
            
            # Add grid aligned with data points
            axs[i].grid(True, which='minor', axis='y', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            axs[i].grid(False, which='major')  # Turn off major grid
            
            # Customize x-axis ticks to show time divisions
            
            time_labels = [f'{int(24*i/n_intervals):02d}:00' for i in range(n_intervals+1)]
            if i == len(columns)-1:  # Only show time labels on bottom subplot
                axs[i].set_xticklabels(time_labels, rotation=45)
            else:
                axs[i].set_xticklabels([])
            
            # Set y-axis ticks and labels
            axs[i].set_yticks(np.arange(matrix.shape[0]))
            y_labels = [str(date) for date in daily_matrix.index]
            axs[i].set_yticklabels(y_labels)

        axs[-1].set_xlabel('Time of Day')
        plt.suptitle(f'{name} Data for {weeks} weeks (Weeks {start_period%52}-{end_period%52})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
if __name__ == '__main__':
    # Load the dataframes
    #dataframes = pd.read_pickle('standardized_datasets/Tidepool_SAP100_prev/train_dataframes.pkl')
    #participant = 'train_24b7a7a140092b0d2b1c4754efdd06a832493da1a9af866b304d16113de7abeb'
    #dataframes = pd.read_pickle('standardized_datasets/Ohio2018/train_dataframes.pkl')
    #participant = '588-ws-training'
    dataframes = pd.read_pickle('standardized_datasets/T1DEXI/train_dataframes.pkl')
    participant = '11_TIDEXI_adults'
    
    # Process and plot the data
    process_and_plot(dataframes, participant, weeks=2, interpolate=True, remove_patches=False, apply_thresholds=False, simulate_smbg=False)