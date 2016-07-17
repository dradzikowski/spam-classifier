from reader.AbstractReader import AbstractReader, MODULE_DIR

import os
import logging

logging.basicConfig(
    filename='logs/enron.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CORPUS_DIR = os.path.join(MODULE_DIR, 'data/corpus/enron')


class EnronReader(AbstractReader):
    def read(self):
        pass
