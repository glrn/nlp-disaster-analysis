from sklearn import svm
from sklearn.cross_validation import train_test_split

import classifier
import common
import numpy

from classifier import BagOfWords, svm_fitter
from dataset_parser import Dataset

TEST_SLICE = 0.1

DATASET_PATH =      'dataset/chime-annotation-tweets-DFE-extended.csv'
POS_TAGGING_PATH =  'dataset/chime-annotation-tweets-DFE-POS-Tagging.txt'
NER_TAGGING_PATH =  'dataset/chime-annotation-tweets-DFE-NER-tags.txt'

def setup():
    print('Starting...')
    print('Parsing dataset...')
    dataset = Dataset()
    print('Done parsing, dataset length: {}'.format(len(dataset.entries)))

    print('Parsing annotated dataset...')
    annotated_dataset = Dataset(DATASET_PATH, POS_TAGGING_PATH, NER_TAGGING_PATH)
    print('Done parsing, annotated dataset length: {}'.format(len(annotated_dataset.entries)))

    print('Splitting into train {} and test {}'.format(1 - TEST_SLICE, TEST_SLICE))
    train, test = train_test_split(dataset.entries, test_size=TEST_SLICE)
    annotated_train, annotated_test = train_test_split(annotated_dataset.entries, test_size=TEST_SLICE)

    return train, test, annotated_train, annotated_test


def test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, **kwds):
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, **kwds)

    print('Fitting...')
    bag.fit_forest(n_estimators=100)
    bag.fit_naive_bayes()

    print('Predicting...')
    print('FOREST:')
    result = bag.predict_forest(test_corpus)
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))

    print('NAIVE BAYES:')
    result = bag.predict_naive_bayes(test_corpus)
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))

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
    tested = svm_fitter(test)
    # You need to play with this C value to get better accuracy (for example if C=1, all predictions are 0).
    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    print('Predicting...')
    result = svm_classifier.predict(tested)
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))

def test_svm_annotated(train, test):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.relevance for tweet in train])
    test_labels = numpy.array([tweet.relevance for tweet in test])

    print('SVM:')
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, ngram_range=(1, 2))
    classifier.vocabulary = bag.vocabulary

    print('Fitting...')
    trained = svm_fitter(train)
    tested = svm_fitter(test)
    # You need to play with this C value to get better accuracy (for example if C=1, all predictions are 0).
    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    print('Predicting...')
    result = svm_classifier.predict(tested)
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))

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


    train, test, annotated_train, annotated_test = setup()

    """
    train_corpus = numpy.array([tweet.text for tweet in train])
    test_corpus = numpy.array([tweet.text for tweet in test])
    train_labels = numpy.array([tweet.label for tweet in train])
    test_labels = numpy.array([tweet.label for tweet in test])
    print('===============================')
    print('Test unigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels)
    print('===============================')
    print('Test unigrams and bigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, ngram_range=(1, 2))
    """
    print('===============================')
    print('Test SVM unigrams and bigrams:')
    test_svm(train, test)


    print('===============================')
    print('Test SVM unigrams and bigrams:')
    test_svm_annotated(annotated_train, annotated_test)




if __name__ == '__main__':
    main()
