from abc import ABC, abstractmethod

class Dataloader(ABC):
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.all_dataframes = {}
        self.train_dataframes = {}
        self.test_dataframes = {}
        self.train_metadata = {}
        self.test_metadata = {}

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod     
    def get_metadata(self):
        pass

    @abstractmethod
    def _get_dataframe(self, file):
        pass
    
    @abstractmethod
    def _get_dataset_specific_metadata(self):
        pass

    def get_metadata(self):
        self._get_dataset_specific_metadata()

        for key, df in self.train_dataframes.items():
            if key not in self.train_metadata:
                self.train_metadata[key] = {}
            self.train_metadata[key].update(self.update_metadata(df))
    
        for key, df in self.test_dataframes.items():
            if key not in self.test_metadata:
                self.test_metadata[key] = {}
            self.test_metadata[key].update(self.update_metadata(df))

    def update_metadata(self, df):
        desc_stats = df['cbg'].describe()
        # Statistical Features
        #mean_glucose = df['cbg'].mean()
        #variance_glucose = df['cbg'].var()
        #covariance_glucose = df['cgm'].cov(df['cbg'])  # Covariance with itself is the variance
        #coef_variation = df['cbg'].std() / mean_glucose
        #skewness_glucose = df['cbg'].skew()
        #kurtosis_glucose = df['cbg'].kurtosis()
        #maximum_glucose = df['cbg'].max()
        #minimum_glucose = df['cbg'].min()
        #slope_glucose = np.polyfit(range(len(df['cbg'])), df['cbg'], 1)[0]
        # Generate descriptive statistics
        # Statistical Features from describe()
        number_of_samples = int(desc_stats['count'].item())
        number_of_cgm_samples = len(df['cbg'])
        mean_glucose = float(desc_stats['mean'].item())
        std_dev_glucose = float(desc_stats['std'].item())
        min_glucose = float(desc_stats['min'].item())
        percentile_25 = float(desc_stats['25%'].item())
        median_glucose = float(desc_stats['50%'].item())
        percentile_75 = float(desc_stats['75%'].item())
        max_glucose = float(desc_stats['max'].item())

        # Clinically Relevant Features
        # HbA1c ... not calculated from CGM data
        mean_glucose = mean_glucose  # already calculated
        percent_time_in_range = float((df['cbg'].between(70, 180).mean()) * 100)
        percent_time_tight_range = float((df['cbg'].between(70, 140).mean()) * 100)
        percent_time_low = float((df['cbg'] < 70).mean() * 100)
        percent_time_very_low = float((df['cbg'] < 54).mean() * 100)
        percent_time_high = float((df['cbg'] > 180).mean() * 100)
        percent_time_very_high = float((df['cbg'] > 250).mean() * 100)
        number_hypo_events = int((df['cbg'] < 70).sum())

        # Demographic Features
        # These are not calculated from the CGM data and would require additional demographic information.

        # Store results in a dictionary
        results = {
            #'Mean': mean_glucose,
            #'Variance': variance_glucose,
            #'Covariance': covariance_glucose,
            #'Coefficient of Variation': coef_variation,
            #'Skewness': skewness_glucose,
            #'Kurtosis': kurtosis_glucose,
            #'Maximum': maximum_glucose,
            #'Minimum': minimum_glucose,
            #'Slope': slope_glucose,
            'number_of_samples': number_of_samples,
            'number_of_cgm_samples': number_of_cgm_samples,
            'Mean': mean_glucose,
            'Standard Deviation': std_dev_glucose,
            'Minimum': min_glucose,
            '25th Percentile': percentile_25,
            'Median': median_glucose,
            '75th Percentile': percentile_75,
            'Maximum': max_glucose,
            '% Time in Range': percent_time_in_range,
            '% Time in Tight Range': percent_time_tight_range,
            '% Time Low': percent_time_low,
            '% Time Very Low': percent_time_very_low,
            '% Time High': percent_time_high,
            '% Time Very High': percent_time_very_high,
            '# Hypo Events': number_hypo_events,
            'Features': list(df.columns)
        }
        return results
        # Create a DataFrame to display the results
        #results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
        #print(results_df)