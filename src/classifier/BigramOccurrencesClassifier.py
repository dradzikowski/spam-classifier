from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


def get():
    return Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('classifier', BernoulliNB(binarize=0.0))
    ])
