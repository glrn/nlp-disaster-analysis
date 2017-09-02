import csv
from tweet_parser import Tweet

"""
This module parses CrowdFlower's dataset.
[URL: https://www.crowdflower.com/data-for-everyone (under 'Disasters on social media')]
"""

DATASET_PATH =      'dataset/socialmedia-disaster-tweets-DFE-extended.csv'
POS_TAGGING_PATH =  'dataset/socialmedia-disaster-tweets-DFE-POS-Tagging.txt'

class Dataset(object):
    """
    This object contains all the data on our tweets.
    """
    def __init__(self, dataset_path=DATASET_PATH, pos_tagging_path=POS_TAGGING_PATH):
        self.entries = []

        pos_of_tweets = read_conll_pos_file(pos_tagging_path)

        t = 0
        with open(dataset_path, 'rb') as csvfile:
            for row in csv.DictReader(csvfile):
                # Handle only tweets with confidence > 0.9
                if float(row['choose_one:confidence']) < 0.9:
                    continue
                POS_tags = ' '.join([tup[1] for tup in pos_of_tweets[t]])
                self.entries.append(Tweet(row, POS_tags))
                t += 1


def read_conll_pos_file(path):
    """
    Takes a path to a file and returns a list of word/tag pairs
    (This code is adopted from the exercises)
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                word, tag, acc = line.strip().split("\t")
                curr.append((word,tag))
    return sents