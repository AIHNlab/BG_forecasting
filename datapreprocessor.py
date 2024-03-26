#Merged into DataPrepper (Deprecated)
class DataPreProcessor:
    def __init__(self, dataframes, fill_types=None):
        self.dataframes = dataframes
        if fill_types is None:
            fill_types = ['linear' for _ in range(len(dataframes))]
        self.fill_types = fill_types

    def handle_missing_values(self, channels):
        for channel, fill_type in zip(channels, self.fill_types):
            for df in self.dataframes.values():
                if isinstance(fill_type, str):
                    df[channel].interpolate(method=fill_type, inplace=True)
                    df[channel].fillna(method='ffill', inplace=True)
                    df[channel].fillna(method='bfill', inplace=True)
                else:
                    df[channel].fillna(fill_type, inplace=True)
        for df in self.dataframes.values():
            df.dropna(subset=channels, inplace=True)