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
    true_positive   = 0
    true_negative   = 0
    num_of_pred_pos = len([x for x in prediction if x == 1])
    num_of_pred_neg = len([x for x in prediction if x == 0])
    num_of_real_pos = len([x for x in real if x == 1])
    num_of_real_neg = len([x for x in real if x == 0])

    for i in xrange(len(prediction)):
        p = prediction[i]
        r = real[i]
        if p == r:
            correct += 1
            if p == 0:
                true_negative += 1
            else:
                true_positive += 1

        if corpus is not None and p != r:
            # print false-positives and false-negatives
            #print "Real: %s, Prediction: %s" % (r, p)
            #print "Tweet is: %s" % corpus[i]
            #print
            pass
    sensitivity = float(true_positive) / num_of_real_pos
    specificity = float(true_negative) / num_of_real_neg
    return float(correct) / len(prediction), sensitivity, specificity