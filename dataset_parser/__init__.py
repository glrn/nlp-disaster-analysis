import random
from dataset_parser import Dataset, MAIN_DATASET_PATH, OBJ_SUB_PATH, OBJ_SUB_POS_TAGGING_PATH, MAIN_POS_TAGGING_PATH
from tweet_parser import Tweet
from tweet_parser import Relevancy

if __name__=='__main__':

    # Showcase
    ds = Dataset()

    some_relevant_tweets = [tweet for tweet in ds.entries if \
                            tweet.label == Relevancy.DISASTER and tweet.confidence >= 0.9]
    some_irrelevant_tweets = [tweet for tweet in ds.entries if \
                            tweet.label == Relevancy.NOT_DISASTER and tweet.confidence >= 0.9]


    print '=== Total %d relevant tweets found, for example:' % len(some_relevant_tweets)
    for i in random.sample(range(0, len(some_relevant_tweets)), 5):
        some_relevant_tweets[i].pretty_print()
        print
    print

    print '=== Total %d irrelevant tweets found, for example:' % len(some_irrelevant_tweets)
    for i in random.sample(range(0, len(some_irrelevant_tweets)), 5):
        some_irrelevant_tweets[i].pretty_print()
        print
