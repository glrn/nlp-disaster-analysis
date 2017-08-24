import csv
import numpy
import random
import requests
import twokenizer
from tweet_parser import Tweet
from ttp import ttp
from ttp import utils

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
    def __init__(self, dataset_path=DATASET_PATH):
        self.entries = []

        # TODO: parse POS tags

        with open(DATASET_PATH, 'rb') as csvfile:
            for row in csv.DictReader(csvfile):
                self.entries.append(Tweet(row))
