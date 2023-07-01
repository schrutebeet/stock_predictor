from stock import Stock
from models import Model

apple = Stock('AAPL')
apple.fetch_intraday()
x_train, y_train, x_test, y_test = apple.prepare_train_test_sets(0.8, 60, scale=True)
model = Model(x_train, y_train, x_test, y_test)
print(model.y_test)
