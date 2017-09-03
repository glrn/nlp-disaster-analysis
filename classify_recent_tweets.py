from sklearn import svm
from sklearn.cross_validation import train_test_split
import csv
import classifier
import common
import dataset_parser.tweet_parser
import numpy
from classifier import BagOfWords, svm_fitter
from dataset_parser import Dataset
from shutil import copyfile

TEST_SLICE = 0.1

def main():
    TWEET_CSV_PATH = "dataset/recent_tweets_test/400_chicago_tweets.csv"
    EXTENDED_CSV_PATH = "dataset/recent_tweets_test/400_chicago_tweets-extended.csv"
    POS_TAG_PATH = "dataset/recent_tweets_test/400_chicago_tweets-POS-Tagging.txt"
    NER_TAG_PATH = "dataset/recent_tweets_test/400_chicago_tweets-NER-tags.txt"
    OUT_CSV_PATH = "dataset/recent_tweets_test/400_chicago_tweets-labeled.csv"

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

    lines = csv.DictReader(open(TWEET_CSV_PATH))

    fieldnames = ['timestamp','location','text','choose_one','choose_one:confidence']
    writer = csv.DictWriter(open(OUT_CSV_PATH, 'wb'), fieldnames)
    writer.writeheader()
    i = 0
    for line in lines:
        if labels[i] == 0:
            line["choose_one"] = "Not Relevant"
        else:
            line["choose_one"] = "Relevant"
        writer.writerow(line)
        i += 1



if __name__ == '__main__':
    main()
