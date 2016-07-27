import logging
import os
import pickle
import time

import numpy
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from config.constants import LOGS_DIR
from preprocessor.DataFrameBuilder import DataFrameBuilder
from reader.enron.EnronReader import EnronReader
from reader.trec.TrecReader import TrecReader

PICKLES_DIR = os.path.join('../pickles')
MAIN_LOG = os.path.join(LOGS_DIR, 'main.log')

logging.basicConfig(
    filename=MAIN_LOG,
    level=logging.DEBUG,
    filemode='w',
    format='%(message)s'
)


class Main:
    @staticmethod
    def run():
        logging.info("--------- Start time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        start_time = time.time()
        data = Main.prepare_data()

        classifier = MultinomialNB()
        ngram_range = (1, 2)
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=ngram_range, stop_words='english')),
            ('tfidf_transformer', TfidfTransformer()),
            # ('classifier', BernoulliNB())
            ('classifier', classifier)
        ])
        Main.log_classification_parameters(ngram_range, pipeline)
        logging.info("Data length: " + str(len(data)))
        k_fold = KFold(n=len(data), n_folds=6)
        scores = []
        confusion = numpy.array([[0, 0], [0, 0]])
        confusion = Main.perform_with_cross_validation(confusion, data, k_fold, pipeline, scores, load_from_pickle=False)
        logging.info("--- Execution took: %s seconds ---" % (time.time() - start_time))
        logging.info("--------- End time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        Main.log_results(confusion, data, scores)  # , confusion_matrix)

    @staticmethod
    def perform_with_cross_validation(confusion, data, k_fold, pipeline, scores, load_from_pickle):
        if load_from_pickle:
            pipeline = Main.read_pickle()
        for train_indices, test_indices in k_fold:
            if not load_from_pickle:
                Main.train(data, pipeline, test_indices, train_indices)
            test_features = data.iloc[test_indices]['email'].values
            test_labels = data.iloc[test_indices]['label'].values.astype(str)
            confusion = Main.test(confusion, pipeline, scores, test_features, test_labels,
                                  save_to_pickle=not load_from_pickle)
        return confusion

    @staticmethod
    def test(confusion, pipeline, scores, test_features, test_labels, save_to_pickle):
        predictions = pipeline.predict(test_features)
        Main.log_misclassified_emails(predictions, test_features, test_labels)
        matrix = confusion_matrix(test_labels, predictions)
        logging.info(matrix)
        confusion += matrix
        score = f1_score(test_labels, predictions, pos_label='spam')
        # confusion_matrix = confusion_matrix(test_labels, predictions)
        if save_to_pickle:
            Main.save_pickle(pipeline, score)
        logging.info("Partial score: " + str(score))
        scores.append(score)
        return confusion

    @staticmethod
    def log_misclassified_emails(predictions, test_features, test_labels):
        j = 0
        for i in range(len(predictions)):
            if predictions[i] != test_labels[i]:
                j += 1
                logging.info("###################################################################################")
                logging.info("--- (((" + str(j) + "))) --- was: " + test_labels[i] + ", predicted: " + predictions[i])
                logging.info(test_features[i])
                logging.info("###################################################################################")

    @staticmethod
    def train(data, pipeline, test_indices, train_indices):
        logging.info("Training / test data: " + str(len(train_indices)) + " / " + str(len(test_indices)))
        train_features = data.iloc[train_indices]['email'].values
        train_lables = data.iloc[train_indices]['label'].values.astype(str)
        # fit_transform - learns the vocabulary of the corpus and extracts word count features
        start_time = time.time()
        pipeline.fit(train_features, train_lables)
        end_time = time.time()
        logging.info("Learning took: " + str(end_time - start_time) + " seconds")

    @staticmethod
    def save_pickle(pipeline, score):
        if score > 0.9935:
            with open(os.path.join(PICKLES_DIR, str(score)) + '.pkl', 'wb') as fid:
                pickle.dump(pipeline, fid)

    @staticmethod
    def read_pickle():
        with open(os.path.join(PICKLES_DIR, '0.993612981184.pkl'), 'rb') as fid:
            loaded_pickle = pickle.load(fid)
        return loaded_pickle

    @staticmethod
    def log_results(confusion, data, scores):  # , confusion_matrix):
        print('--------------------------------------')
        # print classifier and settings
        print('Classified: ' + str(len(data)))
        print('Accuracy: ' + str(sum(scores) / len(scores)))
        print('Confusion matrix: \n' + str(confusion))
        plt.matshow(confusion)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        print('--------------------------------------')

    @staticmethod
    def log_classification_parameters(ngram_range, pipeline):
        logging.info("--------------------")
        logging.info("Pipeline steps: ")
        for step in pipeline.steps:
            logging.info("Step: " + step.__str__())
        logging.info("Ngram range: " + ngram_range.__str__())
        logging.info("--------------------")

    @staticmethod
    def run2():
        data = Main.prepare_data()

        count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        counts = count_vectorizer.fit_transform(data['email'].values)
        classifier = MultinomialNB()
        targets = data['label'].values
        classifier.fit(counts, targets)

        examples = ['Congrats! Boss is proud of your promotion. Keep doing well. Regards.',
                    'Congrats! You are lucky one to be offered a promotion!']
        example_counts = count_vectorizer.transform(examples)
        predictions = classifier.predict(example_counts)
        print(predictions)

    @staticmethod
    def prepare_data():
        reader = EnronReader()
        generator = reader.read(100)  # todo
        generator2 = TrecReader().read()
        builder = DataFrameBuilder()
        ## TODO removing stopswords, tokenizing, lemmatization, stemming
        #data = builder.build([generator])
        data = builder.build([generator2])
        #data = builder.build([generator, generator2])
        logging.debug(data.items)
        data = data.reindex(numpy.random.permutation(data.index))
        return data


def main():
    Main.run()


if __name__ == "__main__":
    main()
