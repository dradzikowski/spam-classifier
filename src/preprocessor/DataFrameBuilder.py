from pandas import DataFrame


class DataFrameBuilder:

    def build(self, generators):
        rows = []
        index = []
        for generator in generators:
            for email, label, file in generator:
                rows.append({'email': email, 'label': label})
                index.append(file)

        data_frame = DataFrame(rows, index=index)
        return data_frame
