import os

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODULE_DIR = os.path.abspath(os.path.join('.'))
LOGS_DIR = os.path.join(PROJECT_DIR, '../logs')
CORPUS_DIR = os.path.join(PROJECT_DIR, '../data/corpus')
PICKLE_DIR = os.path.join(PROJECT_DIR, '../pickle')