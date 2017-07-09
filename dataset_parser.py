#!/usr/bin/env python
import csv
import random
import requests
from ttp import ttp
from ttp import utils

"""
This module parses CrowdFlower's dataset.
[URL: https://www.crowdflower.com/data-for-everyone (under 'Disasters on social media')]
"""

DATASET_PATH = 'dataset/socialmedia-disaster-tweets-DFE.csv'
EXPAND_TINYURL_TIMEOUT = 2.0

class Relevancy:
    DISASTER = 1
    NOT_DISASTER = 2

def dataset_as_dict():
    """
    The returned dictionary contains the following keys (as appears on dataset):
        userid                  -
        tweetid                 - CORRUPT, don't use!
        text                    - content of tweet
        location                - (optional)
        keyword                 - e.g.: 'storm', 'suicide%20bombing', 'tsunami'
        choose_one_gold         - UNUSED
        choose_one              - label: either 'Relevant' or 'Not Relevant'
        choose_one:confidence   - confidence of label: between 0 to 1
        _last_judgment_at       - UNIMPORTANT
        _trusted_judgments      - UNIMPORTANT
        _unit_state             - UNIMPORTANT
        _golden                 - UNIMPORTANT
        _unit_id                - unique index
    """
    entries = []
    with open(DATASET_PATH, 'rb') as csvfile:
        for row in csv.DictReader(csvfile):
            # preprocessing on fields
            row['text'] = row['text'].decode('utf-8').encode('ascii', 'replace')
            row['choose_one'] = Relevancy.DISASTER if row['choose_one'] == 'Relevant' else Relevancy.NOT_DISASTER
            row['choose_one:confidence'] = float(row['choose_one:confidence'])
            row['_unit_id'] = int(row['_unit_id'])
            ###
            entries.append(row)
    return entries

def pretty_print_tweet(tweet):
    print '\t%s' % tweet
    p = ttp.Parser()
    ttp_parser = p.parse(tweet)
    print '\t\t Tags in tweet:' + str(ttp_parser.tags)
    print '\t\t Users in tweet:' + str(ttp_parser.users)
    print '\t\t Urls in tweet:' + str(ttp_parser.urls)
    for url in ttp_parser.urls:
        try:
            print '\t\t Following url: ' + ' -> '.join(utils.follow_shortlink(url))
        except requests.RequestException:
            print '\t\t Following url: %s - Timeout' % url
    print

if __name__=='__main__':
    #print utils.follow_shortlinks(['http://t.co/VpGu8z1Lhb'])

    # this is just an example
    all_tweets = dataset_as_dict()

    some_relevant_tweets = [tweet for tweet in all_tweets if \
                            tweet['choose_one'] == Relevancy.DISASTER and tweet['choose_one:confidence'] >= 0.9]
    some_irrelevant_tweets = [tweet for tweet in all_tweets if \
                            tweet['choose_one'] == Relevancy.NOT_DISASTER and tweet['choose_one:confidence'] >= 0.9]


    print '=== Total %d relevant tweets found, for example:' % len(some_relevant_tweets)
    for i in random.sample(range(0, len(some_relevant_tweets)), 5):
        pretty_print_tweet(some_relevant_tweets[i]['text'])
    print

    print '=== Total %d irrelevant tweets found, for example:' % len(some_irrelevant_tweets)
    for i in random.sample(range(0, len(some_irrelevant_tweets)), 5):
        pretty_print_tweet(some_irrelevant_tweets[i]['text'])
