#!/usr/bin/env python
import csv
import random
import requests

"""
This module parses CrowdFlower's dataset.
[URL: https://www.crowdflower.com/data-for-everyone (under 'Disasters on social media')]
"""

DATASET_PATH = 'dataset/socialmedia-disaster-tweets-DFE.csv'
EXPAND_TINYURL_TIMEOUT = 2.0

class Relevancy:
    DISASTER = 1
    NOT_DISASTER = 2

def tinyurl_to_url(tco_url, _timeout = EXPAND_TINYURL_TIMEOUT):
    """
    Twitter uses https://t.co/ to minify URLs in tweets.
    Use this function to get the real URL.
    Default timeout if 0.5s.
    """
    try:
        return requests.get(tco_url, timeout=_timeout).url
    except requests.exceptions.Timeout:
        return None

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
    with open(DATASET_PATH) as csvfile:
        for row in csv.DictReader(csvfile):
            # preprocessing on fields
            row['choose_one'] = Relevancy.DISASTER if row['choose_one'] == 'Relevant' else Relevancy.NOT_DISASTER
            row['choose_one:confidence'] = float(row['choose_one:confidence'])
            row['_unit_id'] = int(row['_unit_id'])
            ###
            entries.append(row)
    return entries

if __name__=='__main__':
    # this is just an example
    all_tweets = dataset_as_dict()

    some_relevant_tweets = [tweet for tweet in all_tweets if \
                            tweet['choose_one'] == Relevancy.DISASTER and tweet['choose_one:confidence'] >= 0.9]
    some_irrelevant_tweets = [tweet for tweet in all_tweets if \
                            tweet['choose_one'] == Relevancy.NOT_DISASTER and tweet['choose_one:confidence'] >= 0.9]

    print '=== Total %d relevant tweets found, for example:' % len(some_relevant_tweets)
    for i in random.sample(range(0, len(some_relevant_tweets)), 5):
        print '\t%s' % some_relevant_tweets[i]['text']
    print
    print '=== Total %d irrelevant tweets found, for example:' % len(some_irrelevant_tweets)
    print '=== for example:'
    for i in random.sample(range(0, len(some_irrelevant_tweets)), 5):
        print '\t%s' % some_irrelevant_tweets[i]['text']

    print
    print '=== Another example - expand tiny urls (timeout is %dsec):' % EXPAND_TINYURL_TIMEOUT
    urls = ['http://t.co/VpGu8z1Lhb', 'http://t.co/3sNyOOhseq', 'http://t.co/Byj5Dfa2rv']
    for url in urls:
        print('\t%s --> %s') % (url, tinyurl_to_url(url))