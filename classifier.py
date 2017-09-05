import common
import numpy
from ttp import ttp
from dataset_parser import pos_tags

from sklearn.ensemble                   import RandomForestClassifier
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.naive_bayes                import BernoulliNB
from feature                            import feature, fitter

from twokenizer import tokenizeRawTweetText

class BagOfWords(object):

    def __init__(self, corpus, labels, **kwds):
        self.vectorizer     = CountVectorizer(analyzer='word', tokenizer=tokenizeRawTweetText, **kwds)
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

# @feature('svm') # 22277
def unigram(inputs):
    corpus = numpy.array([tweet.processed_text for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=vocabulary, analyzer='word', tokenizer=tokenizeRawTweetText)
    return vectorizer.fit_transform(corpus)

@feature('svm') # 22422
def unigram_and_bigram(inputs):
    corpus = numpy.array([tweet.processed_text for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=vocabulary, analyzer='word', tokenizer=tokenizeRawTweetText, ngram_range=(1, 2))
    return vectorizer.fit_transform(corpus)

@feature('svm')
def tweet_meta_features(inputs):
    corpus = numpy.array([tweet.processed_text for tweet in inputs])

    p = ttp.Parser()

    l = []
    for tweet in corpus:
        ttp_parser = p.parse(tweet)

        # List of features
        features = []
        does_tweet_contain_link = (len(ttp_parser.urls) >= 0)
        num_of_links_in_tweet = len(ttp_parser.urls)
        is_twitter_in_links = ('https://twitter.com/' in ttp_parser.urls)
        doest_tweet_contain_userref = (len(ttp_parser.users) >= 0)
        does_tweet_contain_hashtag = (len(ttp_parser.tags) >= 0)
        num_of_hashtags_in_tweet = len(ttp_parser.tags)
        happy_emojy_in_tweet = ('XD' in tweet or ':)' in tweet or
                                '(:' in tweet or '=|' in tweet or
                                '8D' in tweet or ':P' in tweet or
                                ';D' in tweet)

        features = [does_tweet_contain_link,
                    num_of_links_in_tweet,
                    is_twitter_in_links,
                    doest_tweet_contain_userref,
                    does_tweet_contain_hashtag,
                    num_of_hashtags_in_tweet,
                    happy_emojy_in_tweet]

        l.append(features)

    return l

@feature('svm')
def trigram_of_POS_tags(inputs):
    POS_tags_corpus = numpy.array([tweet.POS for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=pos_tags.ALL_POS_TAGS, ngram_range=(1, 3))
    return vectorizer.fit_transform(POS_tags_corpus)


@feature('svm')
def named_entities(inputs):
    named_entities = numpy.array([' '.join(tweet.named_entities) for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 3))
    return vectorizer.fit_transform(named_entities)


@feature('svm')
def hash_tags(inputs):
    hash_tags = numpy.array([' '.join(tweet.hashtags) for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 3))
    return vectorizer.fit_transform(hash_tags)


'''
    let's say you want to add another feature extraction for svm, do as following:
    @feature('svm')
    def foo(corpus):
        return {build_matrix some how} # this matrix should be of (len(corpus) X #num_of_features) dimension.
'''

def svm_fitter(inputs):
    return fitter('svm', inputs)