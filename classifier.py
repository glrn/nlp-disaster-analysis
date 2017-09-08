import common
import numpy

from ttp                                import ttp
from sklearn.ensemble                   import RandomForestClassifier
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.feature_selection          import SelectKBest
from sklearn.naive_bayes                import BernoulliNB
from feature                            import feature, fitter
from twokenizer                         import tokenizeRawTweetText

class BagOfWords(object):

    def __init__(self, corpus, labels, feature_selection=0.8, **kwds):
        self.vectorizer     = CountVectorizer(analyzer='word', tokenizer=tokenizeRawTweetText, **kwds)
        self.bag_of_words   = self.vectorizer.fit_transform(corpus)
        selector            = SelectKBest(k=int(self.bag_of_words.shape[1] * feature_selection))
        selector.fit(self.bag_of_words, labels)
        self.selected       = selector.get_support()
        self.labels         = labels
        self.vocabulary     = self.vectorizer.get_feature_names()
        self.kwds           = kwds

    @common.timeit
    def fit_forest(self, n_estimators=10):
        self.forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        self.forest.fit(self.bag_of_words[:, self.selected], self.labels)

    def predict_forest(self, test):
        vectorize   = CountVectorizer(vocabulary=self.vocabulary, **self.kwds)
        bag         = vectorize.fit_transform(test)
        return self.forest.predict(bag[:, self.selected])

    @common.timeit
    def fit_naive_bayes(self):
        self.nb = BernoulliNB()
        self.nb.fit(self.bag_of_words[:, self.selected], self.labels)

    def predict_naive_bayes(self, test):
        vectorize   = CountVectorizer(vocabulary=self.vocabulary, **self.kwds)
        bag         = vectorize.fit_transform(test)
        return self.nb.predict(bag[:, self.selected])

vocabulary = None

@feature('svm_uni_pos')
@feature('svm_uni') # 22277
def unigram(inputs):
    corpus = numpy.array([tweet.processed_text for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=vocabulary, analyzer='word', tokenizer=tokenizeRawTweetText)
    return vectorizer.fit_transform(corpus)

@feature('svm_bi_pos')
@feature('svm_bi') # 22422
def unigram_and_bigram(inputs):
    corpus = numpy.array([tweet.processed_text for tweet in inputs])
    vectorizer = CountVectorizer(vocabulary=vocabulary, analyzer='word', tokenizer=tokenizeRawTweetText, ngram_range=(1, 2))
    return vectorizer.fit_transform(corpus)

@feature('svm_uni')
@feature('svm_uni_pos')
@feature('svm_bi')
@feature('svm_bi_pos')
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

    return numpy.array(l)

INTERESTING_POS_TAGS = [
    'A',
    'V',
    'R',
    'O',
    '^',
    '$',
    'G',
]

def count_pos(inputs, poses):
    def counter(tweet, poses):
        return numpy.array([tweet.count(pos) for pos in poses])
    return numpy.array([counter(tweet, poses) for tweet in inputs])

@feature('svm_uni_pos')
@feature('svm_bi_pos')
def all_pos_count(inputs):
    POS_tags_corpus = numpy.array([tweet.POS for tweet in inputs])
    return count_pos(POS_tags_corpus, INTERESTING_POS_TAGS)

def svm_uni_fitter(inputs):
    return fitter('svm_uni', inputs)

def svm_bi_fitter(inputs):
    return fitter('svm_bi', inputs)

def svm_uni_pos_fitter(inputs):
    return fitter('svm_uni_pos', inputs)

def svm_bi_pos_fitter(inputs):
    return fitter('svm_bi_pos', inputs)