from darts import TimeSeries
from darts.models import LinearRegressionModel
from sklearn.linear_model import LinearRegression
from darts.dataprocessing.transformers import Scaler
import torch
import pickle
import numpy as np

class SciKitLinearRegressionModel:
    def __init__(self, hp_config, model_path, retrain_model=True):
        self.hp_config = hp_config
        self.model = LinearRegression()
        self.model_path = model_path    
        self.retrain_model = retrain_model

    def train(self, train_loader, val_loader=None):
        X_train, y_train = self._dataloader_to_numpy(train_loader)
        self.model.fit(X_train, y_train)
        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def test(self, test_loader, scaler=None):
        if self.retrain_model:
            print('loading model')
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

        X_test, y_test = self._dataloader_to_numpy(test_loader)
        predictions = self.model.predict(X_test)
        
        if scaler:
            predictions = scaler.inverse_transform(predictions)
            y_test = scaler.inverse_transform(y_test)
        predictions = np.expand_dims(predictions, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        return predictions, y_test

    def _dataloader_to_numpy(self, dataloader):
        X_list, y_list = [], []
        for X_batch, y_batch in dataloader:
            X_list.append(X_batch.numpy().reshape(X_batch.shape[0], -1))
            y_list.append(y_batch.numpy().reshape(y_batch.shape[0], -1))
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        return X, y

class DartsLinearRegressionModel:
    def __init__(self, hp_config, model_path, retrain_model=True):
        self.hp_config = hp_config
        self.model = LinearRegressionModel(
            lags=hp_config['feature_window'],
            lags_past_covariates=hp_config['feature_window'],
            output_chunk_length=hp_config['forecast_steps'],
        )
        self.model_path = model_path    
        self.retrain_model = retrain_model

    def train(self, train_loader, val_loader):
        X_train, y_train = self._dataloader_to_timeseries(train_loader)
        self.model.fit(y_train, past_covariates=X_train)
        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def test(self, test_loader, test=1, scaler=None):
        if test:
            print('loading model')
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

        
        X_test, y_test = self._dataloader_to_timeseries(test_loader)
        forecasts = self.model.historical_forecasts(y_test,
                                               past_covariates = X_test,
                                               #future_covariates = series['test']['future'],
                                               forecast_horizon=self.hp_config['forecast_steps'], 
                                               stride=1,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False,
                                               #start=formatter.params["max_length_input"]
                                               )
        predictions = []
        for forecast in forecasts:
            forecast = forecast.values()

            predictions.append(forecast)
        predictions = np.array(predictions)
        trues = []
        for X_batch, batch_y in test_loader:
            batch_y = batch_y.numpy()
            #true = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)
            trues.append(batch_y)
        trues = np.vstack(trues)[59:]
        if scaler:
            predictions = np.expand_dims(scaler.inverse_transform(predictions[:,:,0]),axis=2)
            trues = np.expand_dims(scaler.inverse_transform(trues[:,:,0]),axis=2)
        return predictions, trues

    def _dataloader_to_timeseries(self, dataloader):
        X_list, y_list = [], []
        for X_batch, y_batch in dataloader:
            X_list.append(X_batch.numpy()[:,-1])
            y_list.append(y_batch.numpy()[:,-1])
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        X_series = TimeSeries.from_values(X)
        y_series = TimeSeries.from_values(y)
        return X_series, y_series