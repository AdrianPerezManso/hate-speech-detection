import nltk
import pickle

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

en_stop_words = stopwords.words('english')

vectorizer = pickle.load(open('models/vectorizer.pickle', 'rb'))

def clean_message(message):
    tokenizer = nltk.RegexpTokenizer('&#[0-9]+;|http\S+|@?\w+')
    stemmer = SnowballStemmer('english')
    words_without_punct = tokenizer.tokenize(message)
    words_without_handles = list(filter(lambda word: not word.startswith('@') and not word.startswith('http') and not word.startswith('&#'), words_without_punct))
    return ' '.join([stemmer.stem(x) for x in words_without_handles])

def fit_messages(messages):
    vectorizer.fit(messages)

def vectorize_messages(messages):
    return vectorizer.transform(messages)

def dump_vectorizer(path):
    pickle.dump(vectorizer, open(path, 'wb'))
