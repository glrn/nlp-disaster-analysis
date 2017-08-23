import common

from sklearn.ensemble                   import RandomForestClassifier
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.naive_bayes                import BernoulliNB
from feature                            import feature, fitter

class BagOfWords(object):

    def __init__(self, corpus, labels, **kwds):
        self.vectorizer     = CountVectorizer(**kwds)
        self.bag_of_words   = self.vectorizer.fit_transform(corpus)
        self.labels         = labels
        self.vocabulary     = self.vectorizer.get_feature_names()
        self.kwds           = kwds

    @common.timeit
    def fit_forest(self, n_estimators=10):
        self.forest = RandomForestClassifier(n_estimators=n_estimators)
        self.forest.fit(self.bag_of_words, self.labels)

    def predict_forest(self, test):
        vectorize   = CountVectorizer(vocabulary=self.vocabulary, **self.kwds)
        bag         = vectorize.fit_transform(test)
        return self.forest.predict(bag)

    @common.timeit
    def fit_naive_bayes(self):
        self.nb = BernoulliNB()
        self.nb.fit(self.bag_of_words, self.labels)

    def predict_naive_bayes(self, test):
        vectorize   = CountVectorizer(vocabulary=self.vocabulary, **self.kwds)
        bag         = vectorize.fit_transform(test)
        return self.nb.predict(bag)

vocabulary = None

@feature('svm') # 22277
def unigram(corpus):
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    return vectorizer.fit_transform(corpus)

#@feature('svm') # 22422
def unigram_and_bigram(corpus):
    vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 2))
    return vectorizer.fit_transform(corpus)

'''
    let's say you want to add another feature extraction for svm, do as following:
    @feature('svm')
    def foo(corpus):
        return {build_matrix some how} # this matrix should be of (len(corpus) X #num_of_features) dimension.
'''

def svm_fitter(inputs):
    return fitter('svm', inputs)