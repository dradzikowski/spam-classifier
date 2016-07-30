from pandas import DataFrame

from preprocessor.Preprocessor import Preprocessor


class DataFrameBuilder:

    preprocessor = Preprocessor()

    def build(self, generators):
        rows = []
        index = []
        for generator in generators:
            for email, label, file in generator:
                #email = self.preprocessor.preprocess(email)
                rows.append({'email': email, 'label': label})
                index.append(file)

        data_frame = DataFrame(rows, index=index)
        return data_frame