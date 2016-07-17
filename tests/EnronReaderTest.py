import unittest

from reader.enron.EnronReader import EnronReader


class EnronReaderTest(unittest.TestCase):
    def test_read(self):
        reader = EnronReader()
        reader.read()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
