import sys
import time
from stock import Stock
from models import Model

def timeit(func):
	def wrapper(*args, **kwargs):
		start = time.time()
		# runs the function
		function = func(*args, **kwargs)
		end = time.time()
		print(f'Elapsed time: {(end - start):.2f} seconds')
	return wrapper

class Runner():
    @staticmethod
    def run(stock, fetch_type='fetch_daily', train_size=0.8, rolling_window=60, scale=False):
        print(f'Running framework for {stock.stock_symbol}')
        getattr(stock, fetch_type)()
        stock.prepare_train_test_sets(train_size, rolling_window, scale=scale)
        base_for_model = Model(stock)
        base_for_model.lstm_nn(viz=True)



def main():
    arguments = sys.argv
    try:
        stock = Stock(arguments[1])
    except Exception as err:
        comment = "Please, add a stock symbol argument when running the framework."
        err_message = f"{type(err).__name__}: {str(err)}. {comment}"
        raise Exception(f"{err_message}")
        
    Runner.run(stock, scale=True)

if __name__ == '__main__':
    main()