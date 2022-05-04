import nltk

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

en_stop_words = stopwords.words('english')

vectorizer = CountVectorizer(ngram_range=(1,2), stop_words=en_stop_words)

def clean_message(message):
    tokenizer = nltk.RegexpTokenizer('&#[0-9]+;|http\S+|@?\w+')
    stemmer = SnowballStemmer('english')
    words_without_punct = tokenizer.tokenize(message)
    words_without_handles = list(filter(lambda word: not word.startswith('@') and not word.startswith('http') and not word.startswith('&#'), words_without_punct))
    return ' '.join([stemmer.stem(x) for x in words_without_handles])

def fit_vectorize_messages(messages):
    vectorizer.fit(messages)
    return vectorizer.transform(messages)

def vectorize_message(message):
    return vectorizer.transform(message)
