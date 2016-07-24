import logging
import os

from config.constants import CORPUS_DIR
from reader.AbstractReader import AbstractReader

TREC_DIR = os.path.join(CORPUS_DIR, 'trec')
LIMIT = 100


class TrecReader(AbstractReader):
    def __init__(self):
        pass

    def read(self):
        for trec_dir in os.listdir(TREC_DIR):
            # spam ../data/000/121
            # ham ../data/000/122
            trec_dir_path = os.path.join(TREC_DIR, trec_dir)
            trec_full_dir = os.path.join(trec_dir_path, 'full')
            trec_data_dir = os.path.join(trec_dir_path, 'data')
            index_file = os.path.join(trec_full_dir, 'index')
            index = open(index_file, 'r', encoding="iso-8859-1")
            for line in index.readlines():
                splitted_line = line.split()
                email_file = open(os.path.join(trec_data_dir , splitted_line[1]), 'r', encoding="iso-8859-1")
                try:
                    #email_text = email_file.read()
                    past_header, lines = False, []
                    NEWLINE = '\n'
                    for line in email_file:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    email_text = "".join(lines)
                    yield email_text, splitted_line[0], splitted_line[1]
                except UnicodeDecodeError as e:
                    print(e)
                    logging.warning("Bad encoding: " + str(e))
                finally:
                    email_file.close()


def main():
    reader = TrecReader()
    for email, label, filename in reader.read():
        print(email)
        print(label)
        print(filename)


if __name__ == "__main__":
    main()
