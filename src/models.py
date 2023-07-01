"""
OBJECTIVE OF THIS MODULE
------------------------
training several machine learning models using training historical data. 
It implements the machine learning algorithm, tunes hyperparameters, 
and evaluates the model's performance, as well as outputs the best 
model.
"""
import pandas as pd
import numpy as np
from stock import Stock
import matplotlib.pyplot as plt
# Model packages
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
# import arch

class Model:
    def __init__(self, stock_inst) -> None:
        self.stock_inst = stock_inst

    def lstm_nn(self, viz=True):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(100, return_sequences=True, input_shape=(self.stock_inst.x_train.shape[1], 1)))
        model.add(keras.layers.LSTM(100, return_sequences=False))
        model.add(keras.layers.Dense(25))
        model.add(keras.layers.Dense(1))
        print(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.stock_inst.x_train, self.stock_inst.y_train, batch_size=50, epochs=3)
        self.predictions = model.predict(self.stock_inst.x_test)
        if self.stock_inst.scaler:
            self.predictions = self.stock_inst.scaler.inverse_transform(self.predictions)
        rmse = np.sqrt(np.mean(self.predictions - self.stock_inst.y_test)**2)
        if viz:
            self._viz_predictions()
        return rmse, self.predictions
    
    def arima(self):
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))

    def _viz_predictions(self):
        data = self.stock_inst.data[['close']]
        training_data_len = int(round(data.shape[0] * self.stock_inst.train_size, 0))
        train = data[:training_data_len].copy()
        test = data[training_data_len:].copy()
        test.loc[:, 'preds'] = self.predictions
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.plot(train)
        plt.plot(test[['close', 'preds']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        image_path = f'opt/{self.stock_inst.stock_symbol}_lstm_plot.png'
        plt.savefig(image_path)
        plt.close()
