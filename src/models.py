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
# import pmdarima
# import arch

class Model:
    def __init__(self, x_train, y_train, x_test, y_test ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def lstm_nn(self, x_train, y_train, x_test, y_test, scaler):
        model = Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        print(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=50, epochs=3)
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        return rmse, predictions

    def viz_predictions(df, predictions, train_size):
        data = df[['close']]
        training_data_len = int(round(data.shape[0] * train_size, 0))
        train = data[:training_data_len]
        test = data[training_data_len:]
        test.loc[:, 'preds'] = predictions
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
