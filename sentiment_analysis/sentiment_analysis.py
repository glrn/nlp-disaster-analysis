import numpy
import re
from feature import feature

classifier = 'sentiment_analysis'

DIGIT       = re.compile('\d')
CAP_WORD    = re.compile('[A-Z]+')
CAP_LETTER  = re.compile('[A-Z]')

def count_tokens(inputs, tokens):
    def counter(tweet, tokens):
        return numpy.array([tweet.count(token) for token in tokens])
    vectorize_counter = numpy.vectorize(lambda tweet: counter(tweet, tokens))
    return vectorize_counter(inputs)

def count_patterns(inputs, patterns):
    def counter(tweet, patterns):
        return numpy.array([len(pattern.findall(tweet)) for pattern in patterns])
    vectorize_counter = numpy.vectorize(lambda tweet: counter(tweet, patterns))
    return vectorize_counter(inputs)

def presence_tokens(inputs, tokens):
    def presence(tweet, tokens):
        return numpy.array([1 if token in tweet else 0 for token in tokens])
    vectorize_counter = numpy.vectorize(lambda tweet: presence(tweet, tokens))
    return vectorize_counter(inputs)

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
    # TODO
    pass

@feature(classifier)
def emoticon_presence(inputs):
    # TODO
    pass

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
    # TODO
    pass

@feature(classifier)
def length(inputs):
    vectorize_length = numpy.vectorize(lambda tweet: len(tweet))
    return vectorize_length(inputs)
