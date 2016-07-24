import logging
import mimetypes

from nltk import WordNetLemmatizer, re, PorterStemmer
from nltk.corpus import stopwords

from preprocessor.Token import Token


class Preprocessor:
    def __init__(self):
        pass

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    link_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    @staticmethod
    def tokenize_test(sentence):
        lemmatizer = WordNetLemmatizer()
        print(" ".join([lemmatizer.lemmatize(i, 'v') for i in sentence.split()]))
        stemmer = PorterStemmer()
        print(" ".join([stemmer.stem(i) for i in sentence.split()]))

    def extract_features(self, text):
        features = []
        tokens = text.split()
        tokens = [token for token in tokens if token not in stopwords.words("english")]
        for token in tokens:
            if len(token) < 3:
                # if len(token.translate(None, string.punctuation)) < 3:
                continue
            if token.isdigit():
                features.append(Token.NUMBER)
            elif "." + token in mimetypes.types_map.keys():
                features.append(Token.ATTACHMENT)
            elif self.link_pattern.match(token):
                features.append(Token.LINK)
            elif token.upper() == token:
                features.append(Token.CAPITAL_LETTERS)
                features.append(self.lemmatizer.lemmatize(token, 'v').lower())
                # features.append(porterStemmer.stem(token.translate(None, string.punctuation)).lower())
                #  features.append(self.lemmatizer.lemmatize(token.translate(None, string.punctuation), 'v').lower())
            else:
                # features.append(self.lemmatizer.lemmatize(token, 'v').lower())
                features.append(self.stemmer.stem(token).lower())
                #   features.append(self.lemmatizer.lemmatize(token.translate(None, string.punctuation), 'v').lower())

        logging.debug(features)
        return features


def main():
    preprocessor = Preprocessor()
    preprocessor.extract_features(
        "Elizabeth, New Jersey, when my mother was being raised there in a flat over her father’s grocery store, was an industrial "
        "port a quarter the size of Newark, dominated by the Irish working class and their politicians and the tightly knit parish "
        "life that revolved around the town’s many churches, and though I never heard her complain of having been pointedly ill-treated "
        "in Elizabeth as a girl, it was not until she married and moved to Newark’s new Jewish neighborhood that she discovered the "
        "confidence that led her to become first a PTA “grade mother,” then a PTA vice president in charge of establishing a Kindergarten"
        " Mothers’ Club, and finally the PTA president, who, after attending a conference in Trenton on infantile paralysis, proposed an"
        " annual March of Dimes dance on January 30 – President Roosevelt’s birthday – that was accepted by most schools.")


if __name__ == "__main__":
    main()
