from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get():
    return Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 1), stop_words='english')),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', SVC(kernel='linear', C=1.0))
    ])
