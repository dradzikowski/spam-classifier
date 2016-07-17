import logging
import os

import bs4

from config.constants import CORPUS_DIR, LOGS_DIR
from reader.AbstractReader import AbstractReader

ENRON_LOG = os.path.join(LOGS_DIR, 'enron.log')
ENRON_DIR = os.path.join(CORPUS_DIR, 'enron')
ENRON1_DIR = os.path.join(ENRON_DIR, 'enron1')
LIMIT = 100

logging.basicConfig(
    filename=ENRON_LOG,
    level=logging.DEBUG,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class EnronReader(AbstractReader):
    def read(self):
        print(ENRON1_DIR)
        for email in os.listdir(ENRON1_DIR)[:LIMIT]:
            print(email)
            email_file = open(email, 'r')
            email_text = email_file.read()
            email_file.close()
            try:
                email_text = bs4.UnicodeDammit.detwingle(email_text).decode('utf-8')
            except:
                logging.error("Bad encoding: '{file}', skipping".format(file=email))
                continue
            email_text = email_text.encode('ascii', 'ignore')
            print(email_text)


def main():
    reader = EnronReader()
    reader.read()


if __name__ == "__main__":
    main()
