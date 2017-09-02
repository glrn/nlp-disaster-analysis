import csv
import random
import twython
# from tweet_parser import Tweet
from dataset_parser.dataset_parser import Tweet
from dataset_parser.tweet_parser import Annotations
from progressbar import Progressbar

"""
This module parses CHIME project dataset.
[URL: https://github.com/kevincstowe/chime-annotation]
"""

DATASET_PATH = 'dataset/chime-annotation.csv'
CONSUMER_KEY = 't3rlfX7OWEwmobYOVlFKTEQtC'
CONSUMER_SECRET = 'ICqLi4k8zWOevNGZgPLxS4ZKCU0lVLptiDBFLhRw83wHS8lQvt'
OAUTH_TOKEN = '276094047-VDz8eqtbQ38GE23U2wAAhdMmWrtwjbp75eKbbLOm'
OAUTH_TOKEN_SECRET = 'ETj5aWOYTmzRMKw4tYlvghuyRfZq3IxULEFMd5SmkNXPU'


class AnnotatedDataset(object):
    """
    This object contains all the data on our tweets.
    """
    def __init__(self, dataset_path=DATASET_PATH):
        self.entries = []

        twitter = twython.Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
        pb = Progressbar('Building extended dataset')
        with open(DATASET_PATH, 'rb') as csvfile:
            all_rows = [row for row in csv.reader(csvfile)]
            total_tweets = len(all_rows)
            for i in xrange(total_tweets):
                row = all_rows[i]
                pb.update_progress(i, total_tweets)
                tweet_id = row[0]
                annotation = row[1]
                annotations_split = annotation.split('/')
                relevance = []
                relevance_metadata = []
                for split in annotations_split:
                    if split != 'None':
                        rel, rel_meta = split.split('-')
                        relevance.append(Annotations.__dict__[rel])
                        relevance_metadata.append(rel_meta)
                    else:
                        relevance.append(Annotations.none)
                try:
                    raw_tweet = twitter.show_status(id=tweet_id)
                    raw_tweet['_unit_id'] = tweet_id
                    raw_tweet['choose_one'] = 'Relevant' if annotation != 'None' else 'NotRelevant'
                    raw_tweet['choose_one:confidence'] = 1 if annotation != 'None' else 0
                    index = random.choice(range(len(relevance)))
                    metadata = relevance_metadata[index] if len(relevance_metadata) > 0 else ''
                    tweet = Tweet(raw_tweet, [], relevance[index], metadata)
                    self.entries.append(tweet)
                except Exception:
                    pass
