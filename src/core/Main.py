import logging
from time import time

import numpy
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config.constants import PICKLE_DIR
from preprocessor.DataFrameBuilder import DataFrameBuilder
from reader.enron.EnronReader import EnronReader
from reader.trec.TrecReader import TrecReader

logging.basicConfig(
    level=logging.DEBUG,
    filemode='w',
    format='%(message)s'
)


class Main:
    @staticmethod
    def run():
        data = self.prepare_data()

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf_transformer', TfidfTransformer()),
            # ('classifier', BernoulliNB())
            ('classifier', MultinomialNB())
        ])
        k_fold = KFold(n=len(data), n_folds=6)
        scores = []
        confusion = numpy.array([[0, 0], [0, 0]])
        for train_indices, test_indices in k_fold:
            train_features = data.iloc[train_indices]['email'].values
            train_lables = data.iloc[train_indices]['label'].values.astype(str)

            test_features = data.iloc[test_indices]['email'].values
            test_labels = data.iloc[test_indices]['label'].values.astype(str)

            # fit_transform - learns the vocabulary of the corpus and extracts word count features
            start_time = time.time()
            logging.info("Started learning... " + start_time)
            pipeline.fit(train_features, train_lables)
            end_time = time.time()
            logging.info("Learning took: " + end_time - start_time)
            from sklearn.externals import joblib
            joblib.dump(pipeline, PICKLE_DIR + end_time + '.pkl')
            # clf = joblib.load('filename.pkl')

            predictions = pipeline.predict(test_features)

            confusion += confusion_matrix(test_labels, predictions)
            score = f1_score(test_labels, predictions, pos_label='spam')
            scores.append(score)

        print('--------------------------------------')
        # print classifier and settings
        print('Classified: ' + str(len(data)))
        print('Accuracy: ' + str(sum(scores) / len(scores)))
        print('Confusion matrix: \n' + str(confusion))
        print('--------------------------------------')

    def run2(self):
        data = self.prepare_data()

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

    def prepare_data(self):
        reader = EnronReader()
        generator = reader.read(100)
        generator2 = TrecReader().read()
        builder = DataFrameBuilder()
        # data = builder.build([generator, generator2])
        data = builder.build([generator])
        logging.debug(data.items)
        data = data.reindex(numpy.random.permutation(data.index))
        return data


def main():
    Main.run()


if __name__ == "__main__":
    main()
