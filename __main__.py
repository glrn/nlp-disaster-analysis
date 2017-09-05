from sklearn import svm
from sklearn.cross_validation import train_test_split

import classifier
import common
import numpy
import os

from classifier                 import BagOfWords, svm_fitter
from dataset_parser             import Dataset, MAIN_DATASET_PATH, OBJ_SUB_PATH, OBJ_SUB_POS_TAGGING_PATH, MAIN_POS_TAGGING_PATH
from sentiment_analysis         import sentiment_analysis_classifier
from sklearn.feature_selection  import SelectKBest
from sklearn.ensemble           import RandomForestClassifier

TEST_SLICE = 0.1
GRAPHS_DIR = 'graphs'

def setup(dataset_path=MAIN_DATASET_PATH, pos_tag_path=MAIN_POS_TAGGING_PATH):
    print('Starting...')
    print('Parsing dataset...')
    dataset = Dataset(dataset_path=dataset_path, pos_tag_path=pos_tag_path)
    print('Done parsing, dataset length: {}'.format(len(dataset.entries)))

    print('Splitting into train {} and test {}'.format(1 - TEST_SLICE, TEST_SLICE))
    train, test = train_test_split(dataset.entries, test_size=TEST_SLICE, random_state=0)

    return train, test

def test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, n_estimators, **kwds):
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, **kwds)

    random_forest_accuracies = []
    print('FOREST:')
    for n_estimator in n_estimators:
        print('Fitting {}...'.format(n_estimator))
        bag.fit_forest(n_estimators=n_estimator)
        print('Predicting...')
        result = bag.predict_forest(test_corpus)
        accuracy = common.compute_accuracy(result, test_labels, test_corpus)
        print('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
        random_forest_accuracies.append(accuracy)

    print('NAIVE BAYES:')
    print('Fitting...')
    bag.fit_naive_bayes()
    print('Predicting...')
    result = bag.predict_naive_bayes(test_corpus)
    naive_bayes_accuracy = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}, ppv: {}, npv: {}'.format(naive_bayes_accuracy.acc, naive_bayes_accuracy.ppv, naive_bayes_accuracy.npv))

    return random_forest_accuracies, naive_bayes_accuracy

def test_svm(train, test, Cs):
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
    accs = []
    for C in Cs:
        print('C={}'.format(C))
        svm_classifier = svm.SVC(C=C)
        svm_classifier.fit(trained, train_labels)

        print('Predicting...')
        result = svm_classifier.predict(tested)

        accuracy = common.compute_accuracy(result, test_labels, test_corpus)
        print('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))

    return accs

@common.timeit
def test_sentiment_analysis(train, test, n_estimators, C):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.objective for tweet in train])
    test_labels = numpy.array([tweet.objective for tweet in test])

    print('Fitting...')
    trained = sentiment_analysis_classifier(train)
    tested  = sentiment_analysis_classifier(test)

    random_forest_accuracies    = []
    svm_accuracies              = []
    for i in range(1, trained.shape[1] + 1):
        print('#features: {}'.format(i))
        selector = SelectKBest(k=i)
        cur_trained = selector.fit_transform(trained, train_labels)
        selected = selector.get_support()
        cur_tested = tested[:, selected]

        print('Random forest:')
        print('Fitting...')
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        forest.fit(cur_trained, train_labels)

        print('Predicting...')
        result = forest.predict(cur_tested)
        accuracy = common.compute_accuracy(result, test_labels, test_corpus)
        print('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
        random_forest_accuracies.append(accuracy)

        print('SVM:')
        print('Fitting...')
        svm_classifier = svm.SVC(C=C, random_state=0)
        svm_classifier.fit(cur_trained, train_labels)

        print('Predicting...')
        result = svm_classifier.predict(cur_tested)
        accuracy = common.compute_accuracy(result, test_labels, test_corpus)
        print('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
        svm_accuracies.append(accuracy)

    return random_forest_accuracies, svm_accuracies

def test_disaster_classification(n_estimators, Cs):
    train, test = setup()
    train_corpus = numpy.array([tweet.text for tweet in train])
    test_corpus = numpy.array([tweet.text for tweet in test])
    train_labels = numpy.array([tweet.label for tweet in train])
    test_labels = numpy.array([tweet.label for tweet in test])
    print('===============================')
    print('Test unigrams:')
    uni_random_forest_accuracies, uni_naive_bayes_accuracy = test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, n_estimators)
    print('===============================')
    print('Test unigrams and bigrams:')
    bi_random_forest_accuracies, bi_naive_bayes_accuracy = test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, n_estimators, ngram_range=(1, 2))

    log_n_estimators = numpy.log2(n_estimators)
    common.plot(
        xs          = [log_n_estimators for _ in range(6)],
        ys          = [
            [acc.acc for acc in uni_random_forest_accuracies],
            [acc.ppv for acc in uni_random_forest_accuracies],
            [acc.npv for acc in uni_random_forest_accuracies],
            [acc.acc for acc in bi_random_forest_accuracies],
            [acc.ppv for acc in bi_random_forest_accuracies],
            [acc.npv for acc in bi_random_forest_accuracies],
        ],
        colors      = [
            'bs-',
            'gs-',
            'rs-',
            'bo-',
            'go-',
            'ro-',
        ],
        x_label     = '#estimators (log2)',
        y_label     = 'accuracy',
        func_labels = [
            'unigram accuracy',
            'unigram ppv',
            'unigram npv',
            'bigram accuracy',
            'bigram ppv',
            'bigram npv',
        ],
        title       = 'Random Forest',
        save        = os.path.join(GRAPHS_DIR, 'random_forest_unigram_vs_bigram_features.png')
    )
    return
    print('===============================')
    print('Test SVM unigrams and bigrams:')
    #test_svm(train, test, Cs)

def test_sentiment_analysis_classification():
    train, test = setup(dataset_path=OBJ_SUB_PATH, pos_tag_path=OBJ_SUB_POS_TAGGING_PATH)
    print('===============================')
    print('Test sentiment analysis:')
    random_forest_accs, svm_accs = test_sentiment_analysis(train, test, n_estimators=10, C=1000)

def main():
    n_estimators = [2**i for i in range(11)]
    Cs           = [10**i for i in range(1, 8)]
    test_disaster_classification(n_estimators, Cs)
    #test_sentiment_analysis_classification()

if __name__ == '__main__':
    main()
