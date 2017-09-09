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
    print('Geographical Entity - ', sorted(Counter(geo_entities).items(), key=lambda x: x[1], reverse=True))
    print('Organization - ', sorted(Counter(org_entities).items(), key=lambda x: x[1], reverse=True))
    print('Person - ', sorted(Counter(per_entities).items(), key=lambda x: x[1], reverse=True))
    print('Geopolitical Entity - ', sorted(Counter(gpe_entities).items(), key=lambda x: x[1], reverse=True))
    print('Time Indicator -', sorted(Counter(tim_entities).items(), key=lambda x: x[1], reverse=True))
    print('Artifact -', sorted(Counter(art_entities).items(), key=lambda x: x[1], reverse=True))
    print('Event -', sorted(Counter(eve_entities).items(), key=lambda x: x[1], reverse=True))
    print('Natural Phenomenon - ', sorted(Counter(nat_entities).items(), key=lambda x: x[1], reverse=True))
    entities = geo_entities + org_entities + per_entities + gpe_entities + tim_entities + art_entities + nat_entities + eve_entities
    print('Top 10 Entities - ', filter(lambda z: 'userref' not in z and 'twitter' not in z and z not in ('in', 'on', 'at', 'the'), map(lambda y: y[0], sorted(Counter(entities).items(), key=lambda x: x[1], reverse=True)))[:10])


def extract_named_entity(named_entities_tree, entity_type):
    return map(lambda result: ' '.join(map(lambda inner_result: inner_result.split('/')[0], result[5:][:-1].split())),
               re.findall('\({}.*\)'.format(entity_type), named_entities_tree))