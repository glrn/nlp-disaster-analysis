import numpy
import re
import string

from emoticon import Emoticon_RE
from feature  import feature, fitter

classifier = 'sentiment_analysis'

DIGIT       = re.compile('\d')
CAP_WORD    = re.compile(r'\b[A-Z]+\b')
CAP_LETTER  = re.compile('[A-Z]')
PUNCTUATION = re.compile(r'[{}]'.format(string.punctuation))

INTERESTING_POS_TAGS = [
    'JJ',
    'JJR',
    'JJS',
    'VB',
    'VBD',
    'VBG',
    'VBN',
    'VBZ',
    'VBP',
    'RB',
    'PRP',
    'PRP$',
    'NNP',
    'NNPS',
    'CD',
    'POS',
    'WP',
]

def count_tokens(inputs, tokens):
    def counter(tweet, tokens):
        return numpy.array([tweet.text.count(token) for token in tokens])
    return numpy.array([counter(tweet, tokens) for tweet in inputs])

def count_patterns(inputs, patterns):
    def counter(tweet, patterns):
        return numpy.array([len(pattern.findall(tweet.text)) for pattern in patterns])
    return numpy.array([counter(tweet, patterns) for tweet in inputs])

def presence_tokens(inputs, tokens):
    def presence(tweet, tokens):
        return numpy.array([1 if token in tweet.text else 0 for token in tokens])
    return numpy.array([presence(tweet, tokens) for tweet in inputs])

def count_pos(inputs, poses):
    def counter(tweet, poses):
        return numpy.array([tweet.POS.count(pos) for pos in poses])
    return numpy.array([counter(tweet, poses) for tweet in inputs])

@feature(classifier)
def exclamation_count(inputs):
    return count_tokens(inputs, ['!'])

@feature(classifier)
def exclamation_presence(inputs):
    return presence_tokens(inputs, ['!'])

@feature(classifier)
def question_mark_count(inputs):
    return count_tokens(inputs, ['?'])

@feature(classifier)
def question_mark_presence(inputs):
    return presence_tokens(inputs, ['?'])

@feature(classifier)
def url_presence(inputs):
    return numpy.array([[1] if tweet.urls else [0] for tweet in inputs])

@feature(classifier)
def emoticon_presence(inputs):
    return count_patterns(inputs, [Emoticon_RE])

@feature(classifier)
def digits_count(inputs):
    return count_patterns(inputs, [DIGIT])

@feature(classifier)
def cap_words_count(inputs):
    return count_patterns(inputs, [CAP_WORD])

@feature(classifier)
def cap_letters_count(inputs):
    return count_patterns(inputs, [CAP_LETTER])

@feature(classifier)
def punctuation_marks_and_symbols_count(inputs):
    return count_patterns(inputs, [PUNCTUATION])

@feature(classifier)
def length(inputs):
    return numpy.array([[len(tweet.text)] for tweet in inputs])

@feature(classifier)
def adjective_count(inputs):
    return count_pos(inputs, ['JJ'])

@feature(classifier)
def comparative_adjective_count(inputs):
    return count_pos(inputs, ['JJR'])

@feature(classifier)
def superlative_adjective_count(inputs):
    return count_pos(inputs, ['JJS'])

@feature(classifier)
def base_form_verb_count(inputs):
    return count_pos(inputs, ['VB'])

@feature(classifier)
def past_tense_verb_count(inputs):
    return count_pos(inputs, ['VBD'])

@feature(classifier)
def present_participle_verb_count(inputs):
    return count_pos(inputs, ['VBG'])

@feature(classifier)
def past_participle_verb_count(inputs):
    return count_pos(inputs, ['VBN'])

@feature(classifier)
def third_person_singular_present_verb_count(inputs):
    return count_pos(inputs, ['VBZ'])

@feature(classifier)
def non_third_person_singular_present_verb_count(inputs):
    return count_pos(inputs, ['VBP'])

@feature(classifier)
def adverb_count(inputs):
    return count_pos(inputs, ['RB'])

@feature(classifier)
def personal_pronoun_count(inputs):
    return count_pos(inputs, ['PRP'])

@feature(classifier)
def possessive_pronoun_count(inputs):
    return count_pos(inputs, ['PRP$'])

@feature(classifier)
def singular_proper_noun_count(inputs):
    return count_pos(inputs, ['NNP'])

@feature(classifier)
def plural_proper_noun_count(inputs):
    return count_pos(inputs, ['NNPS'])

@feature(classifier)
def cardinal_number_count(inputs):
    return count_pos(inputs, ['CD'])

@feature(classifier)
def possessive_ending_count(inputs):
    return count_pos(inputs, ['POS'])

@feature(classifier)
def wh_pronoun_count(inputs):
    return count_pos(inputs, ['WP'])

feature(classifier)
def all_pos_count(inputs):
    return count_pos(inputs, INTERESTING_POS_TAGS)

def sentiment_analysis_classifier(inputs):
    return fitter(classifier, inputs)
