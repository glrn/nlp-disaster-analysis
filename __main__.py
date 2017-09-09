from __future__ import print_function

from sklearn import svm
from sklearn.cross_validation import train_test_split

import argparse
import classifier
import common
import numpy
import os

from classifier                 import BagOfWords, svm_uni_fitter, svm_bi_fitter, svm_uni_pos_fitter, svm_bi_pos_fitter
from dataset_parser import Dataset, MAIN_DATASET_PATH, OBJ_SUB_PATH, OBJ_SUB_POS_TAGGING_PATH, MAIN_POS_TAGGING_PATH, \
    Relevancy
from ner.corpus import get_gmb_reader, GMB_PATH
from ner.ner_chunker import NamedEntityChunker, print_named_entity_parse_results
from sentiment_analysis         import sentiment_analysis_classifier
from sklearn.feature_selection  import SelectKBest
from sklearn.ensemble           import RandomForestClassifier
from feature                    import named_features

TEST_SLICE  = 0.1
GRAPHS_DIR  = os.path.join(os.path.dirname(__file__), 'graphs')
DEBUG       = False

def PRINT(*args, **kwds):
    if DEBUG:
        print(*args, **kwds)

def setup(dataset_path=MAIN_DATASET_PATH, pos_tag_path=MAIN_POS_TAGGING_PATH):
    PRINT('Starting...')
    PRINT('Parsing dataset...')
    dataset = Dataset(dataset_path=dataset_path, pos_tag_path=pos_tag_path)
    PRINT('Done parsing, dataset length: {}'.format(len(dataset.entries)))

    PRINT('Splitting into train {} and test {}'.format(1 - TEST_SLICE, TEST_SLICE))
    train, test = train_test_split(dataset.entries, test_size=TEST_SLICE, random_state=0)

    return train, test

def test_bag_of_words(train_corpus, test_corpus, train_labels, test_labels, n_estimators, **kwds):
    PRINT('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, **kwds)

    random_forest_accuracies = []
    PRINT('FOREST:')
    for n_estimator in n_estimators:
        print('Fitting {}...'.format(n_estimator))
        bag.fit_forest(n_estimators=n_estimator)
        PRINT('Predicting...')
        result = bag.predict_forest(test_corpus)
        accuracy = common.compute_accuracy(result, test_labels, test_corpus, debug=DEBUG)
        PRINT('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
        random_forest_accuracies.append(accuracy)

    PRINT('NAIVE BAYES:')
    PRINT('Fitting...')
    bag.fit_naive_bayes()
    PRINT('Predicting...')
    result = bag.predict_naive_bayes(test_corpus)
    naive_bayes_accuracy = common.compute_accuracy(result, test_labels, test_corpus, debug=DEBUG)
    PRINT('acc: {}, ppv: {}, npv: {}'.format(naive_bayes_accuracy.acc, naive_bayes_accuracy.ppv, naive_bayes_accuracy.npv))

    return random_forest_accuracies, naive_bayes_accuracy

def test_svm(train, test, Cs):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.label for tweet in train])
    test_labels = numpy.array([tweet.label for tweet in test])

    PRINT('SVM:')
    PRINT('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, ngram_range=(1, 2))
    classifier.vocabulary = bag.vocabulary

    PRINT('Fitting...')
    uni_trained     = svm_uni_fitter(train)
    uni_tested      = svm_uni_fitter(test)
    bi_trained      = svm_bi_fitter(train)
    bi_tested       = svm_bi_fitter(test)
    uni_pos_trained = svm_uni_pos_fitter(train)
    uni_pos_tested  = svm_uni_pos_fitter(test)
    bi_pos_trained  = svm_bi_pos_fitter(train)
    bi_pos_tested   = svm_bi_pos_fitter(test)

    # Play with this C value to get better accuracy (for example if C=1, all predictions are 0).

    def benchmark_svm(Cs, train, train_labels, test, test_labels, test_corpus):
        accs = []
        for C in Cs:
            print('C={}'.format(C))
            svm_classifier = svm.SVC(C=C)
            svm_classifier.fit(train, train_labels)

            PRINT('Predicting...')
            result = svm_classifier.predict(test)

            accuracy = common.compute_accuracy(result, test_labels, test_corpus, debug=DEBUG)
            PRINT('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
            accs.append(accuracy)
        return accs

    uni_accs        = benchmark_svm(Cs, uni_trained, train_labels, uni_tested, test_labels, test_corpus)
    bi_accs         = benchmark_svm(Cs, bi_trained, train_labels, bi_tested, test_labels, test_corpus)
    uni_pos_accs    = benchmark_svm(Cs, uni_pos_trained, train_labels, uni_pos_tested, test_labels, test_corpus)
    bi_pos_accs     = benchmark_svm(Cs, bi_pos_trained, train_labels, bi_pos_tested, test_labels, test_corpus)

    return uni_accs, bi_accs, uni_pos_accs, bi_pos_accs

@common.timeit
def test_sentiment_analysis(train, test, n_estimators, C):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.objective for tweet in train])
    test_labels = numpy.array([tweet.objective for tweet in test])

    PRINT('Fitting...')
    trained = sentiment_analysis_classifier(train)
    tested  = sentiment_analysis_classifier(test)

    random_forest_accuracies    = []
    svm_accuracies              = []
    selected_features           = []

    for i in range(1, trained.shape[1] + 1):
        print('#features: {}'.format(i))
        selector = SelectKBest(k=i)
        cur_trained = selector.fit_transform(trained, train_labels)
        selected = selector.get_support()
        cur_tested = tested[:, selected]
        selected_features.append(selected)

        PRINT('Random forest:')
        PRINT('Fitting...')
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        forest.fit(cur_trained, train_labels)

        PRINT('Predicting...')
        result = forest.predict(cur_tested)
        accuracy = common.compute_accuracy(result, test_labels, test_corpus, debug=DEBUG)
        PRINT('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
        random_forest_accuracies.append(accuracy)

        PRINT('SVM:')
        PRINT('Fitting...')
        svm_classifier = svm.SVC(C=C, random_state=0)
        svm_classifier.fit(cur_trained, train_labels)

        PRINT('Predicting...')
        result = svm_classifier.predict(cur_tested)
        accuracy = common.compute_accuracy(result, test_labels, test_corpus, debug=DEBUG)
        PRINT('acc: {}, ppv: {}, npv: {}'.format(accuracy.acc, accuracy.ppv, accuracy.npv))
        svm_accuracies.append(accuracy)

    return random_forest_accuracies, svm_accuracies, selected_features

@common.timeit
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

    forest_uni_max_acc_idx, forest_uni_max_ppv_idx, forest_uni_max_npv_idx = common.max_accuracy(uni_random_forest_accuracies)
    print('Forest uni: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        forest_uni_max_acc_idx,
        uni_random_forest_accuracies[forest_uni_max_acc_idx].acc,
        forest_uni_max_ppv_idx,
        uni_random_forest_accuracies[forest_uni_max_ppv_idx].ppv,
        forest_uni_max_npv_idx,
        uni_random_forest_accuracies[forest_uni_max_npv_idx].npv,
    ))
    forest_bi_max_acc_idx, forest_bi_max_ppv_idx, forest_bi_max_npv_idx = common.max_accuracy(bi_random_forest_accuracies)
    print('Forest bi: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        forest_bi_max_acc_idx,
        uni_random_forest_accuracies[forest_bi_max_acc_idx].acc,
        forest_bi_max_ppv_idx,
        uni_random_forest_accuracies[forest_bi_max_ppv_idx].ppv,
        forest_bi_max_npv_idx,
        uni_random_forest_accuracies[forest_bi_max_npv_idx].npv,
    ))

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
        save        = os.path.join(GRAPHS_DIR, 'DisasterClassification', 'random_forest_unigram_vs_bigram_features.png')
    )

    print('===============================')
    print('Test SVM unigrams and bigrams:')
    svm_uni_accs, svm_bi_accs, svm_uni_pos_accs, svm_bi_pos_accs = test_svm(train, test, Cs)
    svm_uni_max_acc_idx, svm_uni_max_ppv_idx, svm_uni_max_npv_idx = common.max_accuracy(svm_uni_accs)
    print('SVM uni: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        svm_uni_max_acc_idx,
        svm_uni_accs[svm_uni_max_acc_idx].acc,
        svm_uni_max_ppv_idx,
        svm_uni_accs[svm_uni_max_ppv_idx].ppv,
        svm_uni_max_npv_idx,
        svm_uni_accs[svm_uni_max_npv_idx].npv,
    ))
    svm_uni_pos_max_acc_idx, svm_uni_pos_max_ppv_idx, svm_uni_pos_max_npv_idx = common.max_accuracy(svm_uni_pos_accs)
    print('SVM uni pos: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        svm_uni_pos_max_acc_idx,
        svm_uni_pos_accs[svm_uni_pos_max_acc_idx].acc,
        svm_uni_pos_max_ppv_idx,
        svm_uni_pos_accs[svm_uni_pos_max_ppv_idx].ppv,
        svm_uni_pos_max_npv_idx,
        svm_uni_pos_accs[svm_uni_pos_max_npv_idx].npv,
    ))
    svm_bi_max_acc_idx, svm_bi_max_ppv_idx, svm_bi_max_npv_idx = common.max_accuracy(svm_bi_accs)
    print('SVM bi: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        svm_bi_max_acc_idx,
        svm_bi_accs[svm_bi_max_acc_idx].acc,
        svm_bi_max_ppv_idx,
        svm_bi_accs[svm_bi_max_ppv_idx].ppv,
        svm_bi_max_npv_idx,
        svm_bi_accs[svm_bi_max_npv_idx].npv,
    ))
    svm_bi_pos_max_acc_idx, svm_bi_pos_max_ppv_idx, svm_bi_pos_max_npv_idx = common.max_accuracy(svm_bi_pos_accs)
    print('SVM bi pos: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        svm_bi_pos_max_acc_idx,
        svm_bi_pos_accs[svm_bi_pos_max_acc_idx].acc,
        svm_bi_pos_max_ppv_idx,
        svm_bi_pos_accs[svm_bi_pos_max_ppv_idx].ppv,
        svm_bi_pos_max_npv_idx,
        svm_bi_pos_accs[svm_bi_pos_max_npv_idx].npv,
    ))

    log_Cs = numpy.log10(Cs)
    common.plot(
        xs=[log_Cs for _ in range(6)],
        ys=[
            [acc.acc for acc in svm_uni_accs],
            [acc.ppv for acc in svm_uni_accs],
            [acc.npv for acc in svm_uni_accs],
            [acc.acc for acc in svm_uni_pos_accs],
            [acc.ppv for acc in svm_uni_pos_accs],
            [acc.npv for acc in svm_uni_pos_accs],
        ],
        colors=[
            'bs-',
            'gs-',
            'rs-',
            'bo-',
            'go-',
            'ro-',
        ],
        x_label='#C (log10)',
        y_label='accuracy',
        func_labels=[
            'uni_accuracy',
            'uni_ppv',
            'uni_npv',
            'uni_pos_accuracy',
            'uni_pos_ppv',
            'uni_pos_npv',
        ],
        title='SVM',
        save=os.path.join(GRAPHS_DIR, 'DisasterClassification', 'svm_uni_features.png')
    )
    common.plot(
        xs=[log_Cs for _ in range(6)],
        ys=[
            [acc.acc for acc in svm_bi_accs],
            [acc.ppv for acc in svm_bi_accs],
            [acc.npv for acc in svm_bi_accs],
            [acc.acc for acc in svm_bi_pos_accs],
            [acc.ppv for acc in svm_bi_pos_accs],
            [acc.npv for acc in svm_bi_pos_accs],
        ],
        colors=[
            'bs-',
            'gs-',
            'rs-',
            'bo-',
            'go-',
            'ro-',
        ],
        x_label='#C (log10)',
        y_label='accuracy',
        func_labels=[
            'bi_accuracy',
            'bi_ppv',
            'bi_npv',
            'bi_pos_accuracy',
            'bi_pos_ppv',
            'bi_pos_npv',
        ],
        title='SVM',
        save=os.path.join(GRAPHS_DIR, 'DisasterClassification', 'svm_bi_features.png')
    )

    best_results = [
        [
            round(uni_naive_bayes_accuracy.acc, 3),
            round(bi_naive_bayes_accuracy.acc, 3),
            round(uni_random_forest_accuracies[forest_uni_max_acc_idx].acc, 3),
            round(bi_random_forest_accuracies[forest_bi_max_acc_idx].acc, 3),
            round(svm_uni_accs[svm_uni_max_acc_idx].acc, 3),
            round(svm_uni_pos_accs[svm_uni_pos_max_acc_idx].acc, 3),
            round(svm_bi_accs[svm_bi_max_acc_idx].acc, 3),
            round(svm_bi_pos_accs[svm_bi_max_acc_idx].acc, 3),
        ],
        [
            round(uni_naive_bayes_accuracy.ppv, 3),
            round(bi_naive_bayes_accuracy.ppv, 3),
            round(uni_random_forest_accuracies[forest_uni_max_ppv_idx].ppv, 3),
            round(bi_random_forest_accuracies[forest_bi_max_ppv_idx].ppv, 3),
            round(svm_uni_accs[svm_uni_max_ppv_idx].ppv, 3),
            round(svm_uni_pos_accs[svm_uni_pos_max_ppv_idx].ppv, 3),
            round(svm_bi_accs[svm_bi_max_ppv_idx].ppv, 3),
            round(svm_bi_pos_accs[svm_bi_max_npv_idx].ppv, 3),
        ],
        [
            round(uni_naive_bayes_accuracy.npv, 3),
            round(bi_naive_bayes_accuracy.npv, 3),
            round(uni_random_forest_accuracies[forest_uni_max_npv_idx].npv, 3),
            round(bi_random_forest_accuracies[forest_bi_max_npv_idx].npv, 3),
            round(svm_uni_accs[svm_uni_max_npv_idx].npv, 3),
            round(svm_uni_pos_accs[svm_uni_pos_max_npv_idx].npv, 3),
            round(svm_bi_accs[svm_bi_max_npv_idx].npv, 3),
            round(svm_bi_pos_accs[svm_bi_max_npv_idx].npv, 3),
        ],
    ]

    common.plot_table(
        title           = 'Best Results',
        cells           = best_results,
        column_names    = [
            'Uni NB',
            'Bi NB',
            'Uni RF',
            'Bi RF',
            'Uni SVM',
            'Uni POS SVM',
            'Bi SVM',
            'Bi POS SVM',
        ],
        row_names       = [
            'accuracy',
            'ppv',
            'npv',
        ],
        save            = os.path.join(GRAPHS_DIR, 'DisasterClassification', 'best_result_table.png'),
    )

def test_sentiment_analysis_classification(n_estimators, C):
    train, test = setup(dataset_path=OBJ_SUB_PATH, pos_tag_path=OBJ_SUB_POS_TAGGING_PATH)
    print('===============================')
    print('Test sentiment analysis:')
    random_forest_accs, svm_accs, selected_features = test_sentiment_analysis(train, test, n_estimators=n_estimators, C=C)

    random_forest_max_acc_idx, random_forest_max_ppv_idx, random_forest_max_npv_idx = common.max_accuracy(random_forest_accs)
    print('Random Forest: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        random_forest_max_acc_idx,
        random_forest_accs[random_forest_max_acc_idx].acc,
        random_forest_max_ppv_idx,
        random_forest_accs[random_forest_max_ppv_idx].ppv,
        random_forest_max_npv_idx,
        random_forest_accs[random_forest_max_npv_idx].npv,
    ))

    print('Random Forest Best {} features: {}'.format(random_forest_max_acc_idx + 1, ', '.join(common.best_feature_names(named_features, 'sentiment_analysis', selected_features[random_forest_max_acc_idx]))))

    svm_max_acc_idx, svm_max_ppv_idx, svm_max_npv_idx = common.max_accuracy(svm_accs)
    print('SVM: Max acc: {}: {}, Max ppv: {}: {}, Max npv: {}: {}'.format(
        svm_max_acc_idx,
        svm_accs[svm_max_acc_idx].acc,
        svm_max_ppv_idx,
        svm_accs[svm_max_ppv_idx].ppv,
        svm_max_npv_idx,
        svm_accs[svm_max_npv_idx].npv,
    ))

    print('SVM Best {} features: {}'.format(svm_max_acc_idx + 1, ', '.join(common.best_feature_names(named_features, 'sentiment_analysis', selected_features[svm_max_acc_idx]))))

    common.plot(
        xs=[[i + 1 for i in range(len(random_forest_accs))] for _ in range(3)],
        ys=[
            [acc.acc for acc in random_forest_accs],
            [acc.ppv for acc in random_forest_accs],
            [acc.npv for acc in random_forest_accs],
        ],
        colors=[
            'bs-',
            'gs-',
            'rs-',
        ],
        x_label='#features',
        y_label='accuracy',
        func_labels=[
            'accuracy',
            'ppv',
            'npv',
        ],
        title='Random Forest (#estimators={})'.format(n_estimators),
        save=os.path.join(GRAPHS_DIR, 'SentimentAnalysis', 'random_forest.png')
    )

    common.plot(
        xs=[[i + 1 for i in range(len(svm_accs))] for _ in range(3)],
        ys=[
            [acc.acc for acc in svm_accs],
            [acc.ppv for acc in svm_accs],
            [acc.npv for acc in svm_accs],
        ],
        colors=[
            'bs-',
            'gs-',
            'rs-',
        ],
        x_label='#features',
        y_label='accuracy',
        func_labels=[
            'accuracy',
            'ppv',
            'npv',
        ],
        title='SVM (C={})'.format(C),
        save=os.path.join(GRAPHS_DIR, 'SentimentAnalysis', 'SVM.png')
    )

    num_of_features_rf  = sorted(list(set([random_forest_max_acc_idx, random_forest_max_ppv_idx, random_forest_max_npv_idx])))
    num_of_features_svm = list(set([svm_max_acc_idx, svm_max_ppv_idx, svm_max_npv_idx]))

    best_acc_results = []
    for x in num_of_features_rf:
        best_acc_results.append(round(random_forest_accs[x].acc, 3))
    for x in num_of_features_svm:
        best_acc_results.append(round(svm_accs[x].acc, 3))
    best_ppv_results = []
    for x in num_of_features_rf:
        best_ppv_results.append(round(random_forest_accs[x].ppv, 3))
    for x in num_of_features_svm:
        best_ppv_results.append(round(svm_accs[x].ppv, 3))
    best_npv_results = []
    for x in num_of_features_rf:
        best_npv_results.append(round(random_forest_accs[x].npv, 3))
    for x in num_of_features_svm:
        best_npv_results.append(round(svm_accs[x].npv, 3))

    best_results = [best_acc_results, best_ppv_results, best_npv_results]

    common.plot_table(
        title='Best Results',
        cells=best_results,
        column_names=['RF ({})'.format(x + 1) for x in num_of_features_rf] + ['SVM ({})'.format(x + 1) for x in num_of_features_svm],
        row_names=[
            'accuracy',
            'ppv',
            'npv',
        ],
        save=os.path.join(GRAPHS_DIR, 'SentimentAnalysis', 'best_result_table.png'),
    )


@common.timeit
def test_named_entity_recognition(gmb_dataset_size):
    dataset = Dataset(dataset_path=OBJ_SUB_PATH, pos_tag_path=OBJ_SUB_POS_TAGGING_PATH).entries
    training_samples = get_gmb_reader(GMB_PATH)
    print('===============================')
    print('Test named entity recognition:')
    chunker = NamedEntityChunker(training_samples[:gmb_dataset_size])
    ner_disaster_tweets = chunker.parse_tweets([tweet for tweet in dataset if tweet.label == Relevancy.DISASTER])
    print_named_entity_parse_results(ner_disaster_tweets)


def main(disaster_classification, sentiment_analysis, named_entity_recognition, output_dir, debug):
    global DEBUG, GRAPHS_DIR
    n_estimators        = [2**i for i in range(11)]
    Cs                  = [10**i for i in range(1, 8)]
    gmb_dataset_size    = 20000
    if output_dir:
        GRAPHS_DIR = output_dir
    if debug:
        DEBUG = debug
    if disaster_classification:
        test_disaster_classification(n_estimators, Cs)
    if sentiment_analysis:
        test_sentiment_analysis_classification(n_estimators=128, C=10**4)
    if named_entity_recognition:
        test_named_entity_recognition(gmb_dataset_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('-d', '--disaster-classification', help='will train and classify tweets dataset as disaster or not', action='store_true')
    parser.add_argument('-s', '--sentiment-analysis', help='will train and classify disaster related tweets dataset as objective or subjective', action='store_true')
    parser.add_argument('-n', '--named-entity-recognition', help='will classify named entities in disaster related tweets dataset', action='store_true')
    parser.add_argument('-o', '--output', help='output directory for graphs')
    parser.add_argument('-a', '--all', help='equivalent to -d -s -n', action='store_true')
    args = parser.parse_args()
    main(
        disaster_classification = args.disaster_classification or args.all,
        sentiment_analysis      = args.sentiment_analysis or args.all,
        named_entity_recognition= args.named_entity_recognition or args.all,
        output_dir              = args.output,
        debug                   = args.verbose,
    )
