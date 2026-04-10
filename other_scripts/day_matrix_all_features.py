import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

dataframes = pd.read_pickle('standardized_datasets/Tidepool_SAP100/train_dataframes.pkl')

num_months = 1  # Example: 1 month

for name, dataframe in dataframes.items():
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    dataframe['date'] = dataframe['time'].dt.date
    dataframe['year_month'] = dataframe['time'].dt.year * 12 + dataframe['time'].dt.month - 1

    available_periods = dataframe['year_month'].unique()
    possible_start_periods = [period for period in available_periods if period <= (max(available_periods) - num_months + 1)]
    if not possible_start_periods:
        print(f"No continuous {num_months}-month period available in the data for {name}.")
        continue
    start_period = random.choice(possible_start_periods)
    
    end_period = start_period + num_months - 1
    filtered_dataframe = dataframe[(dataframe['year_month'] >= start_period) & (dataframe['year_month'] <= end_period)]
    
    # Define the columns to create matrices for
    columns_to_plot = ['cbg', 'carbInput', 'bolus']
    
    # Create subplots with as many rows as there are features to plot and 1 column
    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15), sharex=True)
    
    for i, column in enumerate(columns_to_plot):
        grouped = filtered_dataframe.groupby('date')[column].apply(list)
        max_length = max(grouped.apply(len), default=0)
        
        matrix = grouped.apply(lambda x: x + [np.nan]*(max_length - len(x))).tolist()
        matrix_array = np.array(matrix)
        
        time_labels = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]
        selected_time_labels = time_labels[::12]
        selected_time_positions = np.linspace(0, max_length-1, len(selected_time_labels))
        
        ax = axs[i]
        #cax = ax.imshow(matrix_array, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=400)
        cax = ax.imshow(matrix_array, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f'{column.capitalize()}')
        if i == len(columns_to_plot) - 1:
            ax.set_xlabel('Time of Day')
        ax.set_ylabel('Day')
        ax.set_yticks(range(0, len(matrix_array), 7))
        ax.set_yticklabels([str(grouped.index[j]) for j in range(0, len(matrix_array), 7)])
        ax.set_xticks(selected_time_positions)
        ax.set_xticklabels(selected_time_labels, rotation=45)

        # Set minor tick positions for grid lines
        ax.set_xticks(np.arange(-0.5, max_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix_array), 1), minor=True)
        
        ax.set_yticklabels([str(grouped.index[j]) for j in range(0, len(matrix_array), 7)])
        ax.set_xticklabels(selected_time_labels, rotation=45)
        
        # Customize grid to align with each box
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
        ax.tick_params(which='minor', size=0)  # Hide minor tick marks
        # Create a colorbar for each subplot
        fig.colorbar(cax, ax=ax, orientation='vertical', label='Value', pad=0.01)
    
    start_year, start_month = divmod(start_period + 1, 12)
    end_year, end_month = divmod(end_period + 1, 12)
    #fig.colorbar(cax, ax=axs, orientation='vertical', label='Value', pad=0.01)
    plt.suptitle(f'Day Matrix for {name} ({start_month}/{start_year}-{end_month}/{end_year})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()