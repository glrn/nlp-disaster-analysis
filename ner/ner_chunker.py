from __future__ import print_function
from collections import Iterable, Counter

from nltk import word_tokenize, pos_tag, re
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI, conlltags2tree

from ner_features import features


class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)

        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(train=train_sents, feature_detector=features, **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

    def parse_tweets(self, tweets):
        regex = re.compile('[,#@`:)(\[\]\'%^~=&*+/;<>{}|!?._]|http[,#@`\-:)(\[\]\'%^=&_*+/;<>{}|.!?a-z]*')
        named_entities_tree = ''
        for tweet in tweets:
            text = str.lower(str(tweet.processed_text))
            text = regex.sub('', text)
            current_tree = self.parse(pos_tag(word_tokenize(text)))
            named_entities_tree += str(current_tree)
        return named_entities_tree


def print_named_entity_parse_results(named_entities_tree):
    geo_entities = extract_named_entity(named_entities_tree, 'geo')
    org_entities = extract_named_entity(named_entities_tree, 'org')
    per_entities = extract_named_entity(named_entities_tree, 'per')
    gpe_entities = extract_named_entity(named_entities_tree, 'gpe')
    tim_entities = extract_named_entity(named_entities_tree, 'tim')
    art_entities = extract_named_entity(named_entities_tree, 'art')
    nat_entities = extract_named_entity(named_entities_tree, 'nat')
    eve_entities = extract_named_entity(named_entities_tree, 'eve')
    print('Geographical Entities -', prettify_entities(geo_entities))
    print('Organization Entities -', prettify_entities(org_entities))
    print('Person Entities -', prettify_entities(per_entities))
    print('Geopolitical Entities -', prettify_entities(gpe_entities))
    print('Time Indicator Entities -', prettify_entities(tim_entities))
    print('Artifact Entities -', prettify_entities(art_entities))
    print('Event Entities -', prettify_entities(eve_entities))
    print('Natural Phenomenon Entities -', prettify_entities(nat_entities))
    entities = geo_entities + org_entities + per_entities + gpe_entities + tim_entities + art_entities + nat_entities + eve_entities
    print('Top 10 Entities -', ', '.join(filter(lambda z: 'userref' not in z and z not in ('in', 'on', 'at', 'the'), map(lambda y: y[0], sorted(Counter(entities).items(), key=lambda x: x[1], reverse=True)))[:10]))


def extract_named_entity(named_entities_tree, entity_type):
    return map(lambda result: ' '.join(map(lambda inner_result: inner_result.split('/')[0], result[5:][:-1].split())),
               re.findall('\({}.*\)'.format(entity_type), named_entities_tree))


def prettify_entities(entities):
    sorted_entities = sorted(Counter(entities).items(), key=lambda x: x[1], reverse=True)
    return ', '.join(map(lambda x: '{}-{}'.format(x[0], x[1]), sorted_entities))