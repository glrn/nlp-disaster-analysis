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

def compute_accuracy(prediction, real, corpus = None):
    if len(prediction) != len(real):
        raise ValueError('prediction {} and real {} length should by equal'.format(len(prediction), len(real)))
    correct = 0
    false_positive = 0
    false_negative = 0
    for i in xrange(len(prediction)):
        p = prediction[i]
        r = real[i]
        if p == r:
            correct += 1
        elif p == 0:
            false_negative += 1
        else:
            false_positive += 1

        if corpus is not None and p != r:
            # print false-positives and false-negatives
            print "Real: %s, Prediction: %s" % (r, p)
            print "Tweet is: %s" % corpus[i]
            print
    return float(correct) / len(prediction), false_positive, false_negative