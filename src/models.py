"""
OBJECTIVE OF THIS MODULE
------------------------
training several machine learning models using training historical data. 
It implements the machine learning algorithm, tunes hyperparameters, 
and evaluates the model's performance, as well as outputs the best 
model.
"""
import numpy as np
from stock import Stock
import matplotlib.pyplot as plt
# Model packages
from tensorflow.keras import Sequential, layers
from statsmodels.tsa.arima.model import ARIMA
# import arch

class Model:
    def __init__(self, df, x_train, y_train, x_test, y_test, scaler=False) -> None:
        self.data = df
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler

    def lstm_nn(self, viz=True):
        model = Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        print(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.x_train, self.y_train, batch_size=50, epochs=3)
        self.predictions = model.predict(self.x_test)
        if self.scaler:
            self.predictions = self.scaler.inverse_transform(self.predictions)
        rmse = np.sqrt(np.mean(predictions - self.y_test)**2)
        if viz:
            self._viz_predictions(self.x_train.shape[1])
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

    def _viz_predictions(self, train_size):
        data = self.data[['close']]
        training_data_len = int(round(data.shape[0] * train_size, 0))
        train = data[:training_data_len]
        test = data[training_data_len:]
        test.loc[:, 'preds'] = self.predictions
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.plot(train)
        plt.plot(test[['close', 'preds']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        image_path = 'opt/lstm_plot.png'
        plt.savefig(image_path)
        plt.close()
