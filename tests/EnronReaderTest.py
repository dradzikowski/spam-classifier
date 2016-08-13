import unittest

from reader.enron.EnronReader import EnronReader


class EnronReaderTest(unittest.TestCase):
    def test_read(self):

        corpus = [
            'Congrats', 'Boss', 'is', 'proud', 'of', 'your', 'promotion', 'Keep', 'doing',
            'well', 'Regards', 'You', 'are', 'lucky', 'one', 'to', 'be', 'offered', 'a'
        ]

        reader = EnronReader()
        training_set = reader.read()
        self.assertTrue('spam' in training_set)
        self.assertTrue('ham' in training_set)
        self.assertEqual(len(training_set), 2)
        self.assertEqual(len(training_set['spam']), 1500)
        self.assertEqual(len(training_set['ham']), 3672)

if __name__ == '__main__':
    unittest.main()