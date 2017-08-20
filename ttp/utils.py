#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unwind short-links e.g. bit.ly, t.co etc to their canonical links"""
from __future__ import unicode_literals, print_function
import requests
import ttp

DEFAULT_GET_TIMEOUT = 2.0

def follow_shortlink(shortlink):
    # Follow redirects of shortlink, return dict of resulting URLs
    # For example - input: 'http://t.co/8o0z9BbEMu'
    #               output: ['http://t.co/8o0z9BbEMu', 'http://bbc.in/16dClPF']

    url = shortlink
    request_result = requests.get(url, timeout = DEFAULT_GET_TIMEOUT)
    redirect_history = request_result.history
    # history might look like:
    # (<Response [301]>, <Response [301]>)
    # where each response object has a URL
    all_urls = []
    for redirect in redirect_history:
        all_urls.append(redirect.url)
    # append the final URL that we finish with
    all_urls.append(request_result.url)
    return all_urls


def pretty_print_tweet(tweet):
    print('\t%s' % tweet)
    p = ttp.Parser()
    ttp_parser = p.parse(tweet)
    print('\t\t Tags in tweet:' + str(ttp_parser.tags))
    print('\t\t Users in tweet:' + str(ttp_parser.users))
    print('\t\t Urls in tweet:' + str(ttp_parser.urls))
    for url in ttp_parser.urls:
        try:
            print('\t\t Following url: ' + ' -> '.join(follow_shortlink(url)))
        except requests.RequestException:
            print('\t\t Following url: %s - Timeout' % url)
    print()