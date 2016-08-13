import logging
import time

import numpy
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt

from classifier import SVMClassifier
from utils import LoggingUtil
from utils import PickleUtil

from classifier import BagOfWordsClassifier
from classifier import BigramCountsClassifier
from classifier import BigramFrequenciesClassifier
from classifier import BigramOccurrencesClassifier
from classifier import BigramOccurrencesClassifierV2

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class Classifier(object):
    pipeline = SVMClassifier.get()
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
                                       save_to_pickle=not load_from_pickle, print_roc=False)
        LoggingUtil.log_results(self.confusion, data, self.scores)

    @staticmethod
    def test(confusion, pipeline, scores, test_features, test_labels, save_to_pickle, print_roc):
        predictions = pipeline.predict(test_features)
        if print_roc:
            Classifier.print_roc_curve(predictions, test_labels)
        LoggingUtil.log_misclassified_emails(predictions, test_features, test_labels)
        matrix = confusion_matrix(test_labels, predictions)
        logging.info(matrix)
        confusion += matrix
        score = f1_score(test_labels, predictions, pos_label='spam')
        # TODO ROC curve
        if save_to_pickle:
            PickleUtil.save_pickle(pipeline, score)
        logging.info("Partial score: " + str(score))
        scores.append(score)
        return confusion

    @staticmethod
    def print_roc_curve(predictions, test_labels):
        test_labels_binary = []
        pred_binary = []
        for test_label in test_labels:
            if test_label == 'spam':
                test_labels_binary.append(1)
            else:
                test_labels_binary.append(0)
        for pred in predictions:
            if pred == 'spam':
                pred_binary.append(1)
            else:
                pred_binary.append(0)
        test_labels_binary = numpy.asarray(test_labels_binary)
        pred_binary = numpy.asarray(pred_binary)
        false_positive_rate, recall, thresholds = roc_curve(test_labels_binary, pred_binary)
        roc_auc = auc(false_positive_rate, recall)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %
                                                         roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')
        plt.show()

    @staticmethod
    def train(data, pipeline, test_indices, train_indices):
        logging.info("Training / test data: " + str(len(train_indices)) + " / " + str(len(test_indices)))
        train_features = data.iloc[train_indices]['email'].values
        train_lables = data.iloc[train_indices]['label'].values.astype(str)
        start_time = time.time()
        pipeline.fit(train_features, train_lables)
        #print(pipeline.get_params()['classifier'].class_prior)
        end_time = time.time()
        logging.info("Learning took: " + str(end_time - start_time) + " seconds")
