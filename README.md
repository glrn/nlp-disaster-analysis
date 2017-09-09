# Analysis of Panic and Alert on Social Media
### Final Project // Advanced Methods in Natural Language Processing @TAU // Spring 2017

### 1. Usage

The project may be used for completing one of the three following tasks:
- Classifying tweets as `disaster` or not related.
- Classifying `disaster-related` tweets as `objective` or `subjective`.
- Extract `named-entities` from `disaster-related` tweets.
    
Usage summary:
```
$ python __main__.py --help

usage: __main__.py [-h] [-v] [-d] [-s] [-n] [-o OUTPUT] [-a]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity
  -d, --disaster-classification
                        will train and classify tweets dataset as disaster or
                        not
  -s, --sentiment-analysis
                        will train and classify disaster related tweets
                        dataset as objective or subjective
  -n, --named-entity-recognition
                        will classify named entities in disaster related
                        tweets dataset
  -o OUTPUT, --output OUTPUT
                        output directory for graphs
  -a, --all             equivalent to -d -s -n
```

#### 1.1 Disaster Classification

The `--disaster-related` flag trains `disaster-related` classifiers and test their prediction accuracy while tuning parameters using *grid search* method (penalty constant in *SVM* and number of estimators in *Random Forest*). We split the dataset into train and test sets.
After the *fitting* phase on the train set, the *predicting* phase will output the following:
- Best accuracies in the format of a table.
- Plot comparable graphs of the classifiers (*SVM*, *Random Forest*, *Naive Bayes*).
Output example:
_______________
```
$ python __main__.py --disaster-classification

Measure times for function: test_disaster_classification (2017-09-09 17:12:58)
===============================
Test unigrams:
Fitting 1...
.
.
.
Fitting 1024...
===============================
Test unigrams and bigrams:
Fitting 1...
.
.
.
Fitting 1024...
Forest uni: Max acc: 7: 0.91124260355, Max ppv: 7: 0.945355191257, Max npv: 7: 0.891975308642
Forest bi: Max acc: 9: 0.905325443787, Max ppv: 6: 0.934065934066, Max npv: 9: 0.88379204893
Saving graph into: Projects\nlp-disaster-analysis\graphs\DisasterClassification\random_forest_unigram_vs_bigram_features.png
===============================
Test SVM unigrams and bigrams:
C=10
.
.
.
C=10000000
SVM uni: Max acc: 3: 0.934911242604, Max ppv: 2: 0.961325966851, Max npv: 3: 0.934640522876
SVM uni pos: Max acc: 3: 0.936883629191, Max ppv: 2: 0.94623655914, Max npv: 3: 0.937704918033
SVM bi: Max acc: 3: 0.932938856016, Max ppv: 2: 0.952127659574, Max npv: 3: 0.926045016077
SVM bi pos: Max acc: 3: 0.944773175542, Max ppv: 3: 0.954545454545, Max npv: 3: 0.938511326861
Saving graph into: Projects\nlp-disaster-analysis\graphs\DisasterClassification\svm_uni_features.png
Saving graph into: Projects\nlp-disaster-analysis\graphs\DisasterClassification\svm_bi_features.png
          Uni NB  Bi NB  Uni RF  Bi RF  Uni SVM  Uni POS SVM  Bi SVM  \
accuracy   0.921  0.864   0.911  0.890    0.935        0.937   0.933
ppv        0.977  1.000   0.945  0.969    0.961        0.946   0.952
npv        0.891  0.813   0.892  0.856    0.935        0.938   0.926

          Bi POS SVM
accuracy       0.945
ppv            0.955
npv            0.939
Saving table into: Projects\nlp-disaster-analysis\graphs\DisasterClassification\best_result_table.png
Total running time of test_disaster_classification in seconds: 383
```

#### 1.2 Sentiment Analysis Classification

The `--sentiment-analysis` flag trains `objective/subjective` classifiers and test their prediction accuracy while tuning number of features using feature selection methods. We split the dataset into train and test sets.
After the *fitting* phase on the train set, the *predicting* phase will output the following:
- Best accuracies in the format of a table.
- The selected features which each classifier used when it achieved its best accuracy.
- Plot comparable graphs of the classifiers (*SVM*, *Random Forest*).
Output example:
_______________
```
$ python __main__.py --disaster-classification

===============================
Test sentiment analysis:
Measure times for function: test_sentiment_analysis (2017-09-09 17:31:22)
#features: 1
.
.
.
#features: 18
Total running time of test_sentiment_analysis in seconds: 24
Random Forest: Max acc: 12: 0.871428571429, Max ppv: 0: 0.905063291139, Max npv: 12: 0.75
Random Forest Best 13 features: exclamation_count, exclamation_presence, question_mark_presence, url_presence, emoticon_presence, digits_count, cap_letters_count, punctuation_marks_and_symbols_count, length
SVM: Max acc: 3: 0.847619047619, Max ppv: 0: 0.905063291139, Max npv: 3: 0.666666666667
SVM Best 4 features: exclamation_count, url_presence, emoticon_presence, punctuation_marks_and_symbols_count
Saving graph into: Projects\nlp-disaster-analysis\graphs\SentimentAnalysis\random_forest.png
Saving graph into: Projects\nlp-disaster-analysis\graphs\SentimentAnalysis\SVM.png
          RF (1)  RF (13)  SVM (1)  SVM (4)
accuracy   0.805    0.871    0.805    0.848
ppv        0.905    0.890    0.905    0.874
npv        0.500    0.750    0.500    0.667
Saving table into: Projects\nlp-disaster-analysis\graphs\SentimentAnalysis\best_result_table.png

```

#### 1.3 Named Entity Recognition

The `--named-entity-recognition` .... TODO: Omri fill explanation.
Output example:
_______________
```
$ python __main__.py --named-entity-recognition

TODO: Omri fill output.

```

#### 1.4 Other

For more *verbosed* output, `--verbose` flag can be passed. It will result more debugging prints as well as printing *false positive* and *false negative* tweets.
Example:
```
$ python __main__.py --sentiment-analysis --verbose
.
.
.    
Real: 0, Prediction: 1
Tweet is: Hiroshima: 70 years since the worst mass murder in human history. Never forget. http://www.aljazeera.com/ Atomic bomb in 1945: A look back at the destruction - Al Jazeera English
.
.
.
```
The `--output` flag may be used in order to change plots directory, if not given the default directory is *graphs* inside the project package.

### 2. Project Structure

### 3. Prerequisites
* [Python Twitter Tools](https://pypi.python.org/pypi/twitter) library (```pip install twitter```)

### 4. References
For convenience purposes, we hard copied into the project most of the packages we used. The following list details each and every package we used:
- [ark-tweet-nlp-0.3.2](https://github.com/ianozsvald/ark-tweet-nlp-python) library (*POS tagging*)
- [emoticon](https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py) script (*Emoticons extraction*)
- [twitter-text-python](https://pypi.python.org/pypi/twitter-text-python/) library (*Tweets parsing*)
- [Twitter-API](https://dev.twitter.com/rest/public/search) API (*API for tweets extraction*)
- [twokenizer](https://github.com/ataipale/geotagged_tweet_exploration/blob/master/twokenizer.py) script (*Separating tweets into tokens*)
- NER referneces TODO: Omri fill.