from sklearn import svm
from sklearn.cross_validation import train_test_split

import classifier
import common
from classifier import BagOfWords, svm_fitter
from dataset_parser.dataset_parser import dataset_as_dict, get_corpus, \
    get_labels

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
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))

    print('NAIVE BAYES:')
    result = bag.predict_naive_bayes(test_corpus)
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))

def test_svm(train_corpus, test_courpus, train_labels, test_labels):
    print('SVM:')
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, ngram_range=(1, 2))
    classifier.vocabulary = bag.vocabulary

    print('Fitting...')
    trained = svm_fitter(train_corpus)
    tested  = svm_fitter(test_courpus)
    # You need to play with this C value to get better accuracy (for example if C=1, all predictions are 0).
    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    print('Predicting...')
    result = svm_classifier.predict(tested)
    acc = common.compute_accuracy(result, test_labels, test_courpus)
    print('acc: {}'.format(acc))

def main():

    train_corpus, test_corpus, train_labels, test_labels = setup()
    '''
    print('===============================')
    print('Test unigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels)
    print('===============================')
    print('Test unigrams and bigrams:')
    test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, ngram_range=(1, 2))
    '''
    print('===============================')
    print('Test SVM unigrams and bigrams:')
    test_svm(train_corpus, test_corpus, train_labels, test_labels)

if __name__ == '__main__':
    main()
