import csv
from ttp import ttp

# INPUT_FILE = 'chicago_tweets-extended.csv'
# OUTPUT_FILE = 'chicago_tweets-processed.txt'
# INPUT_FILE = 'houston_tweets-extended.csv'
# OUTPUT_FILE = 'houston_tweets-processed.txt'
INPUT_FILE = 'miami_tweets-extended.csv'
OUTPUT_FILE = 'miami_tweets-processed.txt'

with open(INPUT_FILE, 'rb') as csvfile:
    with open(OUTPUT_FILE, 'wb') as txtfile:
        for row in csv.DictReader(csvfile):
            tweet = ttp.process_tweet(row)
            tweet = tweet.replace('\r', '')
            tweet = tweet.replace('\n', '')
            txtfile.write(tweet)
            txtfile.write('\r\n')
