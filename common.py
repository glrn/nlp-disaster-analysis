import datetime
import time

DATE_FORMAT_STRING = '%Y-%m-%d %H:%M:%S'

def timeit(func):
    def wrapper(*args, **kwds):
        start   = time.time()
        print('Measure times for function: {} ({})'.format(func.__name__, datetime.datetime.now().strftime(DATE_FORMAT_STRING)))
        ret = func(*args, **kwds)
        end     = time.time()
        print('Total running time of {} in seconds: {}'.format(func.__name__, int(end - start)))
        return ret
    return wrapper

def compute_accuracy(prediction, real):
    if len(prediction) != len(real):
        raise ValueError('prediction {} and real {} length should by equal'.format(len(prediction), len(real)))
    correct = 0
    for p, r in zip(prediction, real):
        if p == r:
            correct += 1
    return float(correct) / len(prediction)