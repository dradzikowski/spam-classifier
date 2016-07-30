import logging

import numpy

from preprocessor.DataFrameBuilder import DataFrameBuilder
from reader.enron.EnronReader import EnronReader
from reader.trec.TrecReader import TrecReader


def prepare_data():
    reader = EnronReader()
    enronGenerator = reader.read(500)
    trecGenerator = TrecReader().read(5000)
    builder = DataFrameBuilder()
    #data = builder.build([trecGenerator])
    data = builder.build([enronGenerator, trecGenerator])
    logging.debug(data.items)
    data = data.reindex(numpy.random.permutation(data.index))
    logging.info("Data length: " + str(len(data)))
    return data
