import csv

# INPUT_FILE = 'chicago_tweets.csv'
# OUTPUT_FILE = 'chicago_tweets.txt'
# INPUT_FILE = 'houston_tweets.csv'
# OUTPUT_FILE = 'houston_tweets.txt'
INPUT_FILE = 'miami_tweets.csv'
OUTPUT_FILE = 'miami_tweets.txt'

with open(INPUT_FILE, 'rb') as csvfile:
    with open(OUTPUT_FILE, 'wb') as txtfile:
        for row in csv.DictReader(csvfile):
            tweet = row['text'].decode('utf-8').encode('ascii', 'replace').strip()
            tweet = tweet.replace('\r','')
            tweet = tweet.replace('\n', '')
            txtfile.write(tweet)
            txtfile.write('\r\n')
