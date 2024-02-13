class DataPreProcessor:
    def __init__(self, dataframes):
        self.dataframes = dataframes

    def handle_missing_values(self, channels, fill_types):
        for channel, fill_type in zip(channels, fill_types):
            for df in self.dataframes.values():
                if isinstance(fill_type, str):
                    df[channel].interpolate(method=fill_type, inplace=True)
                    df[channel].fillna(method='ffill', inplace=True)
                    df[channel].fillna(method='bfill', inplace=True)
                else:
                    df[channel].fillna(fill_type, inplace=True)
        for df in self.dataframes.values():
            df.dropna(subset=channels, inplace=True)