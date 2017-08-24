import csv

INPUT_FILE = 'socialmedia-disaster-tweets-DFE.csv'
OUTPUT_FILE = 'socialmedia-disaster-tweets-DFE-only-tweets-LOLOLOLOLOL.txt'

with open(INPUT_FILE, 'rb') as csvfile:
    with open(OUTPUT_FILE, 'wb') as txtfile:
        for row in csv.DictReader(csvfile):
            tweet = row['text'].decode('utf-8').encode('ascii', 'replace').strip()
            tweet = tweet.replace('\r','')
            tweet = tweet.replace('\n', '')
            txtfile.write(tweet)
            txtfile.write('\r\n')
