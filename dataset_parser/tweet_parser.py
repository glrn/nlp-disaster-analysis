from ttp import ttp

class Relevancy(object):
    NOT_DISASTER    = 0
    DISASTER        = 1


class Annotations(object):
    none            = 0
    Information     = 1
    Sentiment       = 2
    Reporting       = 3
    Miscellaneous   = 4
    Actions         = 5
    Preparation     = 6
    Other           = 7
    Movement        = 8


class Tweet(object):
    """
    This object contains a tweet and the corresponding metadata (e.g. tweet
    itself, information about links in tweet, part-of-speech tagging, etc.)
    (e.g. tweet's text,
    """

    def __init__(self, rec, POS_tagging, named_entities=list(), relevance=0, relevance_metadata=''):
        """

        :param rec:         record from csv
        :param POS_tagging: list containing part-of-speech tags
        :param named_entities:  list of named-entities in the tweet (default: empty list)
        """

        # Pasrse record from dataset. Keys in the dataset csv are:
        #     userid                  -
        #     tweetid                 - CORRUPT, don't use!
        #     text                    - content of tweet
        #     location                - (optional)
        #     keyword                 - e.g.: 'storm', 'suicide%20bombing', 'tsunami'
        #     choose_one_gold         - UNUSED
        #     choose_one              - label: either 'Relevant' or 'Not Relevant'
        #     choose_one:confidence   - confidence of label: between 0 to 1
        #     _last_judgment_at       - UNIMPORTANT
        #     _trusted_judgments      - UNIMPORTANT
        #     _unit_state             - UNIMPORTANT
        #     _golden                 - UNIMPORTANT
        #     _unit_id                - unique index
        #     link_url1               - Real URL of 1st tiny-URL (if exists)
        #     link_uri1               - URI (domain)
        #     link_title1             - <title> tag of html
        #     link_url2               - Same for 2nd tiny-URL (if exists)
        #     link_uri2               - ...
        #     link_title2             - ...
        #     link_url3               - Same for 3rd tiny-URL (if exists)
        #     link_uri3               - ...
        #     link_title3             - ...
        self.text = rec['text']
        self.text = self.text.decode('utf-8').encode('ascii', 'replace') # convert to ASCII
        self.objective = int(rec['objective']) if 'objective' in rec else None
        self.processed_text = ttp.process_tweet(rec)
        self.label = Relevancy.DISASTER if rec['choose_one'] == 'Relevant' \
                         else Relevancy.NOT_DISASTER
        self.confidence = float(rec['choose_one:confidence'])
        if '_unit_id' in rec.keys():
            self.id = int(float(rec['_unit_id']))

        # Extract information from tweet with ttp
        p = ttp.Parser()
        ttp_parser = p.parse(self.text)
        self.hashtags = ttp_parser.tags
        self.users = ttp_parser.users
        self.urls = ttp_parser.urls

        # Handle POS tagging
        self.POS = POS_tagging

        # Handle relevance
        self.relevance = relevance
        self.relevance_metadata = relevance_metadata

        # Handle NEs
        self.named_entities = named_entities

    def pretty_print(self):
        print('Original tweet:\t%s' % self.text)
        print('Processed tweet: %s' % self.processed_text)
        print('\t Tags in tweet:' + str(self.hashtags))
        print('\t Users in tweet:' + str(self.users))
        print('\t Urls in tweet:' + str(self.urls))
