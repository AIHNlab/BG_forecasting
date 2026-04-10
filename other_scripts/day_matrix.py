import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Replace 'your_dataframe_path.pkl' with the actual path to your .pkl file
dataframe_path = 'your_dataframe_path.pkl'

# Load the dataframe from the .pkl file
dataframes = pd.read_pickle('standardized_datasets/Tidepool_SAP100/train_dataframes.pkl')

# Print the loaded dataframe
print(dataframes)

num_months = 1  # Example: 3 months

for name, dataframe in dataframes.items():
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    dataframe['date'] = dataframe['time'].dt.date
    dataframe['year_month'] = dataframe['time'].dt.year * 12 + dataframe['time'].dt.month - 1  # Unique period identifier

    # Determine a random start period from the available data, adjusting for the specified number of months
    available_periods = dataframe['year_month'].unique()
    possible_start_periods = [period for period in available_periods if period <= (max(available_periods) - num_months + 1)]
    if not possible_start_periods:
        print(f"No continuous {num_months}-month period available in the data for {name}.")
        continue
    start_period = random.choice(possible_start_periods)
    
    # Filter for the specified number of continuous months
    end_period = start_period + num_months - 1
    filtered_dataframe = dataframe[(dataframe['year_month'] >= start_period) & (dataframe['year_month'] <= end_period)]
    
    grouped = filtered_dataframe.groupby('date')['cbg'].apply(list)
    max_length = max(grouped.apply(len))
    
    matrix = grouped.apply(lambda x: x + [np.nan]*(max_length - len(x))).tolist()
    matrix_array = np.array(matrix)
    
    time_labels = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]
    selected_time_labels = time_labels[::12]
    selected_time_positions = np.linspace(0, max_length-1, len(selected_time_labels))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_array, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=400)
    plt.colorbar(label='CBG')
    # Convert start_period and end_period back to readable format for title
    start_year, start_month = divmod(start_period + 1, 12)
    end_year, end_month = divmod(end_period + 1, 12)
    plt.title(f'Day Matrix for {name} ({start_month}/{start_year}-{end_month}/{end_year})')
    plt.xlabel('Time of Day')
    plt.ylabel('Day')
    plt.yticks(range(0, len(matrix_array), 7), [str(grouped.index[i]) for i in range(0, len(matrix_array), 7)])
    plt.xticks(selected_time_positions, selected_time_labels, rotation=45)
    plt.show()