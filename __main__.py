import numpy as np

import common

from sklearn.cross_validation   import train_test_split

from classifier                 import BagOfWords
from dataset_parser             import dataset_as_dict, get_corpus, get_labels

TEST_SLICE = 0.1

def setup():
    print('Starting...')
    print('Parsing dataset...')
    dataset = dataset_as_dict()
    print('Done parsing, dataset length: {}'.format(len(dataset)))

    print('Splitting into train {} and test {}'.format(1 - TEST_SLICE, TEST_SLICE))
    train, test = train_test_split(dataset, test_size=TEST_SLICE)

    print('Generating corpuses and labels...')
    train_corpus, test_corpus = get_corpus(train), get_corpus(test)
    train_labels, test_labels = get_labels(train), get_labels(test)
    return train_corpus, test_corpus, train_labels, test_labels

def test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, **kwds):
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, **kwds)

    print('Fitting...')
    bag.fit_forest(n_estimators=100)
    bag.fit_naive_bayes()

    print('Predicting...')
    print('FOREST:')
    result = bag.predict_forest(test_corpus)
    acc = common.compute_accuracy(result, test_labels)
    print('acc: {}'.format(acc))

    print('NAIVE BAYES:')
    result = bag.predict_naive_bayes(test_corpus)
    acc = common.compute_accuracy(result, test_labels)
    print('acc: {}'.format(acc))

def main():

    train_corpus, test_corpus, train_labels, test_labels = setup()
    print('===============================')
    print('Test unigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels)
    print('===============================')
    print('Test unigrams and bigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, ngram_range=(1, 2))

if __name__ == '__main__':
    main()
