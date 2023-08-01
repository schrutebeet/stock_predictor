import sys
import time
from stock import Stock
from models import Model

def timeit(func):
    @staticmethod
    def wrapper(*args, **kwargs):
        start = time.time()
        # runs the function
        function = func(*args, **kwargs)
        end = time.time()
        print("-"*25)
        print(f'Elapsed time: {(end - start):.2f} seconds')
        print("-"*25, "\n")
        return function
    return wrapper

class Runner():
    @timeit
    def run(stock, fetch_type='fetch_daily', train_size=0.8, rolling_window=60, scale=True):
        print(f'Running framework for {stock.stock_symbol}')
        getattr(stock, fetch_type)()
        stock.prepare_train_test_sets(train_size, rolling_window, scale=scale)
        base_for_model = Model(stock)
        base_for_model.lstm_nn(viz=True)


def main():
    arguments = sys.argv
    try:
        stock_names = arguments[1]
        print(stock_names)
        stock = Stock(stock_names)
    except Exception as err:
        comment = "Please, add a stock symbol argument when running the framework."
        err_message = f"{type(err).__name__}: {str(err)}. {comment}"
        raise Exception(f"{err_message}")
        
    Runner.run(stock, scale=True)

if __name__ == '__main__':
    main()