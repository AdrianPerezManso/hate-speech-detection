from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from configs import config, logconfig
from utils import file_management as fm
import nltk
import wordsegment as ws
import logging

"""
Preprocessing of text
"""

ws.load()
data = fm.json_to_data(config.INIT_JSON_DIR)
[nltk.download(package, download_dir=config.NLTK_DATA_DIR) for package in data[config.NLTK][config.PACKAGES] if data[config.NLTK][config.DOWNLOAD]]

EN_STOP_WORDS = stopwords.words('english')

def clean_message(message):
    #logging.debug(logconfig.LOG_NLP_START)
    tokenizer = nltk.RegexpTokenizer(config.REGEX_TOKENIZER)
    stemmer = SnowballStemmer(config.LANGUAGE)

    lower_tweet = message.lower()
    #logging.debug(logconfig.LOG_NLP_LOWER)

    tokens = tokenizer.tokenize(lower_tweet)
    #logging.debug(logconfig.LOG_NLP_TOKEN)

    tokens_sent= ' '.join(tokens)
    for word in tokens:
      if word.startswith(config.HASHTAG):
        split_hashtag = ' '.join(ws.segment(word))
        tokens_sent = tokens_sent.replace(word, split_hashtag)
        #logging.debug(logconfig.LOG_NLP_HASHTAG_PROCESSING)

    words_without_handles = list(filter(lambda word: not word.startswith(config.REGEX_AT) and 
                                                     not word.startswith(config.REGEX_HTTP) and 
                                                     not word.startswith(config.REGEX_UNICODE), tokens_sent.split()))
    #logging.debug(logconfig.LOG_NLP_REMOVE_USERNAME_LINK_UNICODE)
    stemmed_text = ' '.join([stemmer.stem(x) if not x in EN_STOP_WORDS else x for x in words_without_handles])
    #logging.debug(logconfig.LOG_NLP_STEM)
    return stemmed_text

def get_stopwords():
    return EN_STOP_WORDS

def get_tokenizer_function():
    return nltk.word_tokenize

