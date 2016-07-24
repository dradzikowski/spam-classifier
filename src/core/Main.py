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

from preprocessor.DataFrameBuilder import DataFrameBuilder
from reader.enron.EnronReader import EnronReader
from reader.trec.TrecReader import TrecReader

PICKLES_DIR = os.path.join('../pickles')

logging.basicConfig(
    level=logging.DEBUG,
    filemode='w',
    format='%(message)s'
)


class Main:
    @staticmethod
    def run():
        data = Main.prepare_data()

        classifier = MultinomialNB()
        ngram_range = (1, 2)
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=ngram_range)),
            ('tfidf_transformer', TfidfTransformer()),
            # ('classifier', BernoulliNB())
            ('classifier', classifier)
        ])
        Main.log_classification_parameters(ngram_range, pipeline)
        logging.info("Data length: " + str(len(data)))
        k_fold = KFold(n=len(data), n_folds=6)
        scores = []
        confusion = numpy.array([[0, 0], [0, 0]])
        for train_indices, test_indices in k_fold:
            logging.info("Training / test data: " + str(len(train_indices)) + " / " + str(len(test_indices)))
            train_features = data.iloc[train_indices]['email'].values
            train_lables = data.iloc[train_indices]['label'].values.astype(str)

            test_features = data.iloc[test_indices]['email'].values
            test_labels = data.iloc[test_indices]['label'].values.astype(str)

            # fit_transform - learns the vocabulary of the corpus and extracts word count features
            start_time = time.time()
            pipeline.fit(train_features, train_lables)
            end_time = time.time()
            logging.info("Learning took: " + str(end_time - start_time) + " seconds")

            predictions = pipeline.predict(test_features)

            matrix = confusion_matrix(test_labels, predictions)
            logging.info(matrix)
            confusion += matrix
            score = f1_score(test_labels, predictions, pos_label='spam')

            if score > 0.92:
                # save the classifier
                with open(os.path.join(PICKLES_DIR, str(score)) + '.pkl', 'wb') as fid:
                    pickle.dump(pipeline, fid)

                    # load it again
                    # with open('my_dumped_classifier.pkl', 'rb') as fid:
                    #    gnb_loaded = pickle.load(fid)

            logging.info("Partial score: " + str(score))
            scores.append(score)

        print('--------------------------------------')
        # print classifier and settings
        print('Classified: ' + str(len(data)))
        print('Accuracy: ' + str(sum(scores) / len(scores)))
        print('Confusion matrix: \n' + str(confusion))
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
        generator = reader.read(100) # todo
        generator2 = TrecReader().read()
        builder = DataFrameBuilder()
        ## TODO removing stopswords, tokenizing, lemmatization, stemming
        data = builder.build([generator])
        #TODO data = builder.build([generator, generator2])
        logging.debug(data.items)
        data = data.reindex(numpy.random.permutation(data.index))
        return data


def main():
    Main.run()


if __name__ == "__main__":
    main()
