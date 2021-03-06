from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def get():
    return Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('classifier', MultinomialNB())
    ])
