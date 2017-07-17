import common

from sklearn.ensemble                   import RandomForestClassifier
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.naive_bayes                import BernoulliNB

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
