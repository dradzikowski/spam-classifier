import logging
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from classifier import classifier
from config.constants import LOGS_DIR
from utils import DataUtil
from utils import LoggingUtil

MAIN_LOG = os.path.join(LOGS_DIR, 'main.log')

logging.basicConfig(
    filename=MAIN_LOG,
    level=logging.DEBUG,
    filemode='w',
    format='%(message)s'
)


class Main:
    @staticmethod
    def run_for_set():
        start_time = LoggingUtil.log_start_time()

        data = DataUtil.prepare_data()
        cls = classifier.Classifier()
        cls.perform_with_cross_validation(data, load_from_pickle=True)

        LoggingUtil.log_end_time(start_time)

    @staticmethod
    def run_for_examples():
        start_time = LoggingUtil.log_start_time()
        data = DataUtil.prepare_data()

        cls = MultinomialNB()
        vect = CountVectorizer(ngram_range=(1, 2))

        train_labels = data['label'].values
        train_features = vect.fit_transform(data['email'].values)
        cls.fit(train_features, train_labels)

        examples = ['Congrats! Boss is proud of your promotion. Keep doing well. Regards.',
                    'Congrats! You are lucky one to be offered a promotion!',
                    'Congrats! You are promoted!',
                    'Congrats! You won one million!']
        test_features = vect.transform(examples)
        predictions = cls.predict(test_features)

        print(predictions)
        LoggingUtil.log_end_time(start_time)


def main():
    Main.run_for_set()


if __name__ == "__main__":
    main()
