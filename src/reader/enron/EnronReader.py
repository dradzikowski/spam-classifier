import json
import logging
import os
from collections import defaultdict

import bs4

from config.constants import CORPUS_DIR, LOGS_DIR
from reader.AbstractReader import AbstractReader

ENRON_LOG = os.path.join(LOGS_DIR, 'enron.log')
ENRON_DIR = os.path.join(CORPUS_DIR, 'enron')
ENRON1_DIR = os.path.join(ENRON_DIR, 'enron1')
HAM_DIR = os.path.join(ENRON1_DIR, 'ham')
SPAM_DIR = os.path.join(ENRON1_DIR, 'spam')
LIMIT = 100

logging.basicConfig(
    #filename=ENRON_LOG,
    level=logging.DEBUG,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class EnronReader(AbstractReader):
    def read(self):
        training_set = defaultdict(list)
        for label in os.listdir(ENRON1_DIR):
            label_dir = os.path.join(ENRON1_DIR, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir)[:LIMIT]:
                    logging.debug("Reading email: [" + label + "] " + file)
                    file_dir = os.path.join(label_dir, file)
                    email_file = open(file_dir, 'r')
                    email_text = email_file.read()
                    email_file.close()
                    training_set[label].append(email_text)
        logging.debug("Training set: " + json.dumps(training_set, indent=4))
        return training_set


def main():
    reader = EnronReader()
    reader.read()


if __name__ == "__main__":
    main()
