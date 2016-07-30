import logging
import time

import numpy
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import LoggingUtil
from utils import PickleUtil

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class Classifier(object):
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('tfidf_transformer', TfidfTransformer()),
        # ('classifier', BernoulliNB())
        ('classifier', MultinomialNB())
    ])
    confusion = numpy.array([[0, 0], [0, 0]])
    scores = []

    def perform_with_cross_validation(self, data, load_from_pickle):
        if load_from_pickle:
            self.pipeline = PickleUtil.read_pickle()
        LoggingUtil.log_classification_parameters(self.pipeline)
        k_fold = KFold(n=len(data), n_folds=6)
        for train_indices, test_indices in k_fold:
            if not load_from_pickle:
                self.train(data, self.pipeline, test_indices, train_indices)
            test_features = data.iloc[test_indices]['email'].values
            test_labels = data.iloc[test_indices]['label'].values.astype(str)
            self.confusion = self.test(self.confusion, self.pipeline, self.scores, test_features, test_labels,
                                       save_to_pickle=not load_from_pickle)
        LoggingUtil.log_results(self.confusion, data, self.scores)

    @staticmethod
    def test(confusion, pipeline, scores, test_features, test_labels, save_to_pickle):
        predictions = pipeline.predict(test_features)
        LoggingUtil.log_misclassified_emails(predictions, test_features, test_labels)
        matrix = confusion_matrix(test_labels, predictions)
        logging.info(matrix)
        confusion += matrix
        score = f1_score(test_labels, predictions, pos_label='spam')
        if save_to_pickle:
            PickleUtil.save_pickle(pipeline, score)
        logging.info("Partial score: " + str(score))
        scores.append(score)
        return confusion

    @staticmethod
    def train(data, pipeline, test_indices, train_indices):
        logging.info("Training / test data: " + str(len(train_indices)) + " / " + str(len(test_indices)))
        train_features = data.iloc[train_indices]['email'].values
        train_lables = data.iloc[train_indices]['label'].values.astype(str)
        start_time = time.time()
        pipeline.fit(train_features, train_lables)
        end_time = time.time()
        logging.info("Learning took: " + str(end_time - start_time) + " seconds")
