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
        cls.perform_with_cross_validation(data, load_from_pickle=False)

        LoggingUtil.log_end_time(start_time)

    @staticmethod
    def run_for_examples():
        start_time = LoggingUtil.log_start_time()
        data = DataUtil.prepare_data()

        count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        counts = count_vectorizer.fit_transform(data['email'].values)
        classifier = MultinomialNB()
        targets = data['label'].values
        classifier.fit(counts, targets)

        examples = ['Congrats! Boss is proud of your promotion. Keep doing well. Regards.',
                    'Congrats! You are lucky one to be offered a promotion!']
        example_counts = count_vectorizer.transform(examples)
        predictions = classifier.predict(example_counts)

        logging.info(predictions)

        LoggingUtil.log_end_time(start_time)


def main():
    Main.run_for_set()


if __name__ == "__main__":
    main()
