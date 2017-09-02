import csv
import time
import StreamingAPI
import SearchAPI

BOSTON_GEOCODE = "42.362393,-71.062971,10km"
CHICAGO_GEOCODE = "41.881832,-87.623177,10km"
ROCKPORT_TEXAS_GEOCODE = "28.048611,-97.041111,10km"
HOUSTON_TEXAS_GEOCODE = "29.789054,-95.387083,10km"

if __name__ == '__main__':
    keywords = ['car','plane','fire','dead']
    with open("chicago_tweets.csv", 'wb') as csvfile:
        fieldnames = ['timestamp', 'location', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for kw in keywords:
            tweets = SearchAPI.fetch(kw, geocode=CHICAGO_GEOCODE, count=100)
            for t in tweets:
                print t
                writer.writerow(t)
            csvfile.flush()

    keywords = ['water','drown','']
    with open("houston_tweets.csv", 'wb') as csvfile:
        fieldnames = ['timestamp', 'location', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for kw in keywords:
            tweets = SearchAPI.fetch(kw, geocode=HOUSTON_TEXAS_GEOCODE, count=100)
            for t in tweets:
                print t
                writer.writerow(t)
