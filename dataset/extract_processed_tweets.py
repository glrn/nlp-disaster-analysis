import csv
from ttp import ttp

INPUT_FILE = 'socialmedia-disaster-tweets-DFE-extended.csv'
OUTPUT_FILE = 'socialmedia-disaster-tweets-DFE-only-tweets-processed.txt'

with open(INPUT_FILE, 'rb') as csvfile:
    with open(OUTPUT_FILE, 'wb') as txtfile:
        for row in csv.DictReader(csvfile):
            tweet = ttp.process_tweet(row)
            tweet = tweet.replace('\r', '')
            tweet = tweet.replace('\n', '')
            txtfile.write(tweet)
            txtfile.write('\r\n')
