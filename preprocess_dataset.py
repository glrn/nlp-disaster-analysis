# NOTE: This script creates the pre-processed dataset, hence should be run just once!

# The script extends the dataset by adding information on links (URLs) in tweets.
# For every tweet that includes a link (e.g. http://t.co/3ImaomknnA) we reveal
# the original URL (e.g. https://www.facebook.com/walkercotoday/posts/9894095),
# and fetch the HTML title tag from that web page.
# We handle the first 3 links in every tweet (4th+ link is ignored).

# The processed dataset is an extension of the original dataset, with 9 new
# fields: link_url1, link_uri1, link_title1,
#         link_url2, link_uri2, link_title2,
#         link_url3, link_uri3, link_title3,

import csv
import urllib2
from progressbar import Progressbar
from ttp import utils
from ttp import ttp
from bs4 import BeautifulSoup
from urlparse import urlparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ORIGINAL_DATASET_PATH =  'dataset/socialmedia-disaster-tweets-DFE.csv'
# PROCESSED_DATASET_PATH = 'dataset/socialmedia-disaster-tweets-DFE-extended.csv'
# ORIGINAL_DATASET_PATH =  'dataset/recent_tweets_test/chicago_tweets.csv'
# PROCESSED_DATASET_PATH = 'dataset/recent_tweets_test/chicago_tweets-extended.csv'
# ORIGINAL_DATASET_PATH =  'dataset/recent_tweets_test/houston_tweets.csv'
# PROCESSED_DATASET_PATH = 'dataset/recent_tweets_test/houston_tweets-extended.csv'
ORIGINAL_DATASET_PATH =  'dataset/recent_tweets_test/miami_tweets.csv'
PROCESSED_DATASET_PATH = 'dataset/recent_tweets_test/miami_tweets-extended.csv'

# Read original dataset
logger.info('Reading from %s' % ORIGINAL_DATASET_PATH)
orig_dataset = [tweetRec for tweetRec in csv.DictReader(open(ORIGINAL_DATASET_PATH, 'rb'))]
total_tweets = len(orig_dataset)
logger.info('%d rows fetched' % total_tweets)

# Write new dataset
logger.info('Writing to %s' % PROCESSED_DATASET_PATH)
pb = Progressbar('Building extended dataset')
with open(PROCESSED_DATASET_PATH, 'wb') as csvfile:
    dataset_keys = orig_dataset[0].keys() + ['link_url1', 'link_uri1', 'link_title1',\
                                             'link_url2', 'link_uri2', 'link_title2',\
                                             'link_url3', 'link_uri3', 'link_title3']
    csvwriter = csv.DictWriter(csvfile, fieldnames=dataset_keys)
    csvwriter.writeheader()

    for i in xrange(len(orig_dataset)):
        pb.update_progress(i, total_tweets)

        tweet = orig_dataset[i]['text']

        try:
            # Step 1 - Check if there are any links in tweet
            tweet_parser = ttp.Parser().parse(tweet)

            for j in range(1,4):
                # Step 2 - Follow URL redirection until final URL is revealed
                if len(tweet_parser.urls) >= j:
                    shortened_url = tweet_parser.urls[j-1]
                    logger.info('Following URL %s' % shortened_url)
                    redirected_url = utils.follow_shortlink(shortened_url)[-1]
                    orig_dataset[i]['link_url%d' % j] = unicode(redirected_url).encode("utf-8")

                    parsed_uri = urlparse(redirected_url)
                    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
                    orig_dataset[i]['link_uri%d' % j] = unicode(domain).encode("utf-8")

                    # Step 3 - Get HTML title from the URL
                    soup = BeautifulSoup(urllib2.urlopen(redirected_url), features='lxml')
                    orig_dataset[i]['link_title%d' % j] = unicode(soup.title.string).encode("utf-8")

        except:
            logger.error('catch')
            pass

        # Step 4 - Write new record to the new dataset
        csvwriter.writerow(orig_dataset[i])
