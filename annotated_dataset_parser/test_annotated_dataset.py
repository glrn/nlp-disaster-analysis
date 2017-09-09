def test_svm_annotated(train, test):
    train_corpus = numpy.array([tweet.processed_text for tweet in train])
    test_corpus = numpy.array([tweet.processed_text for tweet in test])
    train_labels = numpy.array([tweet.relevance for tweet in train])
    test_labels = numpy.array([tweet.relevance for tweet in test])

    print('SVM:')
    print('Generating bag of words...')
    bag = BagOfWords(train_corpus, train_labels, ngram_range=(1, 2))
    classifier.vocabulary = bag.vocabulary

    print('Fitting...')
    trained = svm_fitter(train)
    tested = svm_fitter(test)
    # You need to play with this C value to get better accuracy (for example if C=1, all predictions are 0).
    svm_classifier = svm.SVC(C=1000)
    svm_classifier.fit(trained, train_labels)

    print('Predicting...')
    result = svm_classifier.predict(tested)
    acc = common.compute_accuracy(result, test_labels, test_corpus)
    print('acc: {}'.format(acc))
