import pywren

if __name__ == "__main__":
    wrenexec = pywren.default_executor()

    def increment(x):
        return x+1

    x = [1, 2, 3, 4]
    futures = wrenexec.map(increment, x)

    def reduce_func(x):
        return sum(x)

    reduce_future = wrenexec.reduce(reduce_func, futures)
    print reduce_future.result()
