import collections
import functools
import numpy as np
import scipy.sparse

named_features = collections.defaultdict(list)

def feature(name):
    '''
        decorator for specific classifier, the wrapped function should accept as an input a corpus and return a matrix of vector features.
    '''
    def feature_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)
        named_features[name].append(wrapper)
        return wrapper
    return feature_wrapper

def fitter(name, inputs):
    '''
        The fitter function takes all features function that `name` owns and run them one by one on the inputs (corpus).
        Each of those functions return a matrix, which they are all concatenated into one big matrix (concatenated by extending lines).
    '''
    matrices = []
    for f in named_features[name]:
        matrices.append(f(inputs))
    a = matrices[0]
    for b in matrices[1:]:
        try:
            a = np.concatenate((a, b), axis=1)
        except:
            a = scipy.sparse.hstack([a,b])
    return a

