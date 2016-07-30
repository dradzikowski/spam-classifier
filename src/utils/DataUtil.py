import logging

import numpy

from preprocessor.DataFrameBuilder import DataFrameBuilder
from reader.enron.EnronReader import EnronReader
from reader.trec.TrecReader import TrecReader


def prepare_data():
    reader = EnronReader()
    generator = reader.read(100)  # todo
    generator2 = TrecReader().read()
    builder = DataFrameBuilder()
    ## TODO removing stopswords, tokenizing, lemmatization, stemming
    data = builder.build([generator])
    # data = builder.build([generator2])
    # data = builder.build([generator, generator2])
    logging.debug(data.items)
    data = data.reindex(numpy.random.permutation(data.index))
    logging.info("Data length: " + str(len(data)))
    return data
