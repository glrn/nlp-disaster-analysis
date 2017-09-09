from sklearn import svm
from sklearn.cross_validation import train_test_split
import csv
import classifier
import collections
import common
import dataset_parser.tweet_parser
import numpy

from ner.ner_chunker import print_named_entity_parse_results
from classifier import BagOfWords, svm_bi_pos_fitter as svm_fitter
from dataset_parser import Dataset
from shutil import copyfile

from ner.corpus import get_gmb_reader
from ner.ner_chunker import NamedEntityChunker

TEST_SLICE = 0.1

def main():
    # TWEET_CSV_PATH = "dataset/recent_tweets_test/chicago_tweets.csv"
    # EXTENDED_CSV_PATH = "dataset/recent_tweets_test/chicago_tweets-extended.csv"
    # POS_TAG_PATH = "dataset/recent_tweets_test/chicago_tweets-POS-Tagging.txt"
    # NER_TAG_PATH = "dataset/recent_tweets_test/chicago_tweets-NER-tags.txt"
    # OUT_CSV_PATH = "dataset/recent_tweets_test/chicago_tweets-labeled.csv"
    # TWEET_CSV_PATH = "dataset/recent_tweets_test/houston_tweets.csv"
    # EXTENDED_CSV_PATH = "dataset/recent_tweets_test/houston_tweets-extended.csv"
    # POS_TAG_PATH = "dataset/recent_tweets_test/houston_tweets-POS-Tagging.txt"
    # NER_TAG_PATH = "dataset/recent_tweets_test/houston_tweets-NER-tags.txt"
    # OUT_CSV_PATH = "dataset/recent_tweets_test/houston_tweets-labeled.csv"
    TWEET_CSV_PATH = "dataset/recent_tweets_test/miami_tweets.csv"
    EXTENDED_CSV_PATH = "dataset/recent_tweets_test/miami_tweets-extended.csv"
    POS_TAG_PATH = "dataset/recent_tweets_test/miami_tweets-POS-Tagging.txt"
    NER_TAG_PATH = "dataset/recent_tweets_test/miami_tweets-NER-tags.txt"
    OUT_CSV_PATH = "dataset/recent_tweets_test/miami_tweets-labeled.csv"

    labeled_dataset = Dataset()
    unlabeled_dataset = Dataset(dataset_path = EXTENDED_CSV_PATH,
                                pos_tag_path = POS_TAG_PATH,
                                ner_tag_path = NER_TAG_PATH,
                                min_confidence = 0)

    # Train SVM
    train_corpus = numpy.array([tweet.processed_text for tweet in labeled_dataset.entries])
    train_labels = numpy.array([tweet.label for tweet in labeled_dataset.entries])
    bag = BagOfWords(train_corpus, train_labels, ngram_range=(1, 2))
    classifier.vocabulary = bag.vocabulary
    trained = svm_fitter(labeled_dataset.entries)
    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    # Use the trained SVM to label the unlabeled tweets
    tested = svm_fitter(unlabeled_dataset.entries)
    labels = svm_classifier.predict(tested)

    lines = list(csv.DictReader(open(TWEET_CSV_PATH)))

    fieldnames = ['timestamp','location','text','choose_one','choose_one:confidence']
    writer = csv.DictWriter(open(OUT_CSV_PATH, 'wb'), fieldnames)
    writer.writeheader()

    training_samples = get_gmb_reader('ner\gmb-2.2.0')
    chunker = NamedEntityChunker(training_samples[:10000])
    tweets = [unlabeled_dataset.entries[i] for i in xrange(len(lines)) if labels[i] == 1]
    ner_disaster_tweets = chunker.parse_tweets(tweets)
    print_named_entity_parse_results(ner_disaster_tweets)


if __name__ == '__main__':
    main()
