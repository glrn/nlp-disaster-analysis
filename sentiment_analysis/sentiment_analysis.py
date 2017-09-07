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
    'A',
    'V',
    'R',
    'O',
    '^',
    '$',
    'G',
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

'''
@feature(classifier) # 'JJ', 'JJR', 'JJS'
def adjective_count(inputs):
    return count_pos(inputs, ['A'])

@feature(classifier) # 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP'
def verb_count(inputs):
    return count_pos(inputs, ['V'])

@feature(classifier) # 'RB'
def adverb_count(inputs):
    return count_pos(inputs, ['R'])

@feature(classifier) # 'PRP', 'WP'
def wh_and_personal_pronoun_count(inputs):
    return count_pos(inputs, ['O'])

@feature(classifier) # 'NNP', 'NNPS'
def proper_noun_count(inputs):
    return count_pos(inputs, ['^'])

@feature(classifier) # 'CD'
def cardinal_number_count(inputs):
    return count_pos(inputs, ['$'])

@feature(classifier) # 'POS'
def possessive_ending_count(inputs):
    return count_pos(inputs, ['G'])
'''

@feature(classifier)
def all_pos_count(inputs):
    return count_pos(inputs, INTERESTING_POS_TAGS)

def sentiment_analysis_classifier(inputs):
    return fitter(classifier, inputs)
