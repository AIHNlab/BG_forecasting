import math
import matplotlib.pyplot as plt
import seaborn as sns

class Analyser:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def plot_histograms(self):
        df = self.data_handler.get_combined_df()
        df = df.drop(columns='5minute_intervals_timestamp')
        quantiles = df.quantile([0.25, 0.75])
        Q1 = quantiles.loc[0.25]
        Q3 = quantiles.loc[0.75]
        IQR = Q3 - Q1

        num_columns = 3
        fig, axs = plt.subplots(math.ceil(len(df.columns) / num_columns), num_columns, figsize=(15, 15))

        for i, column in enumerate(df.columns):
            axs.flat[i].hist(df[column].dropna(), range=(Q1[column] - 1.5*IQR[column], Q3[column] + 1.5*IQR[column]), bins=30)
            axs.flat[i].set_title(column)

        fig.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        df = self.data_handler.get_combined_df()

        # Calculate the correlation matrix
        corr_matrix = df.corr()

        # Plot the heatmap of the correlation matrix
        plt.figure(figsize=(7,7))  # Adjust the size of the figure
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()