import nltk
import pickle
import wordsegment as ws

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from configs import config

EN_STOP_WORDS = stopwords.words('english')
ws.load()
def clean_message(message):
    tokenizer = nltk.RegexpTokenizer(config.REGEX_TOKENIZER)
    stemmer = SnowballStemmer(config.LANGUAGE)
    lower_tweet = message.lower()
    tokens = tokenizer.tokenize(lower_tweet)
    tokens_sent= ' '.join(tokens)
    for word in tokens:
      if word.startswith('#'):
        split_hashtag = ' '.join(ws.segment(word))
        tokens_sent = tokens_sent.replace(word, split_hashtag)
    words_without_handles = list(filter(lambda word: not word.startswith(config.AT) and not word.startswith(config.HTTP) and not word.startswith(config.UNICODE), tokens_sent.split()))
    return ' '.join([stemmer.stem(x) if not x in EN_STOP_WORDS else x for x in words_without_handles])

def get_stopwords():
    return EN_STOP_WORDS

def get_tokenizer_function():
    return nltk.word_tokenize

