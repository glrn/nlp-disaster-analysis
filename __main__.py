from sklearn import svm
from sklearn.cross_validation import train_test_split

import classifier
import common
import dataset_parser.tweet_parser
import numpy
from classifier import BagOfWords, svm_fitter
from dataset_parser import Dataset, MAIN_DATASET_PATH, OBJ_SUB_PATH, OBJ_SUB_POS_TAGGING_PATH, MAIN_POS_TAGGING_PATH
from sentiment_analysis import sentiment_analysis_classifier
TEST_SLICE = 0.1

def setup(dataset_path=MAIN_DATASET_PATH, pos_tag_path=MAIN_POS_TAGGING_PATH):
    print('Starting...')
    print('Parsing dataset...')
    dataset = Dataset(dataset_path=dataset_path, pos_tag_path=pos_tag_path)
    print('Done parsing, dataset length: {}'.format(len(dataset.entries)))

    print('Splitting into train {} and test {}'.format(1 - TEST_SLICE, TEST_SLICE))
    train, test = train_test_split(dataset.entries, test_size=TEST_SLICE, random_state=0)

    return train, test

def test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, **kwds):
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, **kwds)

    print('Fitting...')
    bag.fit_forest(n_estimators=100)
    bag.fit_naive_bayes()

    print('Predicting...')
    print('FOREST:')
    result = bag.predict_forest(test_corpus)
    acc, false_positive, false_negative = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}, #false positive: {}, #false_negative: {}'.format(acc, false_positive, false_negative))

    print('NAIVE BAYES:')
    result = bag.predict_naive_bayes(test_corpus)
    acc, false_positive, false_negative = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}, #false positive: {}, #false_negative: {}'.format(acc, false_positive, false_negative))

def test_svm(train, test):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.label for tweet in train])
    test_labels = numpy.array([tweet.label for tweet in test])

    print('SVM:')
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, ngram_range=(1, 2))
    classifier.vocabulary = bag.vocabulary

    print('Fitting...')
    trained = svm_fitter(train)
    tested  = svm_fitter(test)
    # Play with this C value to get better accuracy (for example if C=1, all predictions are 0).
    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    print('Predicting...')
    result = svm_classifier.predict(tested)
    acc, false_positive, false_negative = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}, #false positive: {}, #false_negative: {}'.format(acc, false_positive, false_negative))

@common.timeit
def test_sentiment_analysis(train, test):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.objective for tweet in train])
    test_labels = numpy.array([tweet.objective for tweet in test])

    print('Fitting...')
    trained = sentiment_analysis_classifier(train)
    tested  = sentiment_analysis_classifier(test)

    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    print('Predicting...')
    result = svm_classifier.predict(tested)
    acc, false_positive, false_negative = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}, #false positive: {}, #false_negative: {}'.format(acc, false_positive, false_negative))

def main():
    #Print some named-entities for relevant tweets
    # ds = Dataset()
    # for tweet in ds.entries[:30]:
    #     if tweet.confidence >= 0.9 and tweet.label == dataset_parser.tweet_parser.Relevancy.DISASTER\
    #             and len(tweet.named_entities) > 0:
    #         print tweet.text
    #         print tweet.processed_text
    #         print "Named Entities: " + str(tweet.named_entities)
    #         print


    train, test = setup(dataset_path=OBJ_SUB_PATH, pos_tag_path=OBJ_SUB_POS_TAGGING_PATH)
    train_corpus = numpy.array([tweet.text for tweet in train])
    test_corpus = numpy.array([tweet.text for tweet in test])
    train_labels = numpy.array([tweet.label for tweet in train])
    test_labels = numpy.array([tweet.label for tweet in test])
    """
    print('===============================')
    print('Test unigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels)
    print('===============================')
    print('Test unigrams and bigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, ngram_range=(1, 2))
    print('===============================')
    print('Test SVM unigrams and bigrams:')
    test_svm(train, test)
    """

    print('===============================')
    print('Test sentiment analysis:')
    test_sentiment_analysis(train, test)



if __name__ == '__main__':
    main()
