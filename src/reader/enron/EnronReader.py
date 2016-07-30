import logging
import os

from config.constants import CORPUS_DIR, LOGS_DIR
from reader.AbstractReader import AbstractReader

ENRON_LOG = os.path.join(LOGS_DIR, 'enron.log')
ENRON_DIR = os.path.join(CORPUS_DIR, 'enron')
LIMIT = 100


class EnronReader(AbstractReader):
    def __init__(self):
        pass

    def read(self, limit_for_subdir=LIMIT):
        for enron_dir in os.listdir(ENRON_DIR):
            for label in os.listdir(os.path.join(ENRON_DIR, enron_dir)):
                label_dir = os.path.join(os.path.join(ENRON_DIR, enron_dir), label)
                if os.path.isdir(label_dir):
                    for file in os.listdir(label_dir)[:limit_for_subdir]:
                        logging.debug("Reading email: [" + label + "] " + file)
                        file_dir = os.path.join(label_dir, file)
                        email_file = open(file_dir, 'r', encoding="iso-8859-1")
                        try:
                            email_text = email_file.read()
                            yield email_text, label, file_dir
                        except UnicodeDecodeError as e:
                            logging.warning("Bad encoding: " + str(e))
                        finally:
                            email_file.close()


def main():
    reader = EnronReader()
    reader.read()


if __name__ == "__main__":
    main()
