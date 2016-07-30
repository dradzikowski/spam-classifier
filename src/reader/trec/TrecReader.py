import email
import logging
import os

from config.constants import CORPUS_DIR
from reader.AbstractReader import AbstractReader

TREC_DIR = os.path.join(CORPUS_DIR, 'trec')
LIMIT = 100


class TrecReader(AbstractReader):
    def __init__(self):
        pass

    def read(self, limit=LIMIT):
        for trec_dir in os.listdir(TREC_DIR):
            trec_dir_path = os.path.join(TREC_DIR, trec_dir)
            trec_full_dir = os.path.join(trec_dir_path, 'full')
            trec_data_dir = os.path.join(trec_dir_path, 'data')
            index_file = os.path.join(trec_full_dir, 'index')
            index = open(index_file, 'r', encoding="iso-8859-1")
            for line in index.readlines()[:limit]:
                splitted_line = line.split()
                email_file = open(os.path.join(trec_data_dir, splitted_line[1]), 'r', encoding="iso-8859-1")
                try:
                    logging.info("Reading email: [" + splitted_line[0] + "] " + splitted_line[1])
                    email_text = email_file.read()
                    email_message = email.message_from_string(email_text)
                    email_text = email_message.get_payload()
                    yield from self.yield_emails(email_text, splitted_line, trec_data_dir)
                except UnicodeDecodeError as e:
                    print(e)
                    logging.warning("Bad encoding: " + str(e))
                finally:
                    email_file.close()

    def yield_emails(self, email_text, splitted_line, trec_data_dir):
        if type(email_text) == list:
            for email_text_part in email_text:
                payload = email_text_part.get_payload()
                self.yield_emails(payload, splitted_line, trec_data_dir)
        # for e in email_text:
        #        yield e, splitted_line[0], os.path.join(trec_data_dir, splitted_line[1])
        else:
            yield email_text, splitted_line[0], os.path.join(trec_data_dir, splitted_line[1])


def main():
    reader = TrecReader()
    for email, label, filename in reader.read():
        print(label)
        print(filename)
        pass
        # print("----------------------- Payload -----------------------")
        # print(email)
        # print(label)
        # print(filename)


if __name__ == "__main__":
    main()
