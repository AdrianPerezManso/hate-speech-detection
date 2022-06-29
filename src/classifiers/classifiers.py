import pickle
import pandas as pd
import os

from configs import config
from abc import ABC, abstractmethod
from joblib import load
from domain.prediction import BinaryPrediction, MLPrediction
from utils import nlp, stats, file_management as fm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    @abstractmethod
    def predict(self, msg: str):
        pass
    @abstractmethod
    def fit_new_data(self, data):
        pass
    @abstractmethod
    def _transform_data(self, data):
        pass

class BinaryModel(Model):
    def __init__(self, train=False):
        if(train): 
            self._train()
        else:
            with open(config.BINARY_MODEL_DIR, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(config.BINARY_VECT_DIR, 'rb') as f:
                self.vectorizer = pickle.load(f)

    def predict(self, msg: str):
        result = self.classifier.predict(self._transform_data(msg))[0]
        return BinaryPrediction(msg, result)
    
    def fit_new_data(self, data):
        x_new, y_new = data.x, data.y
        self.vectorizer.fit(x_new)
        X_new = self._transform_data(x_new)
        self.classifier.fit(X_new, y_new)
        with open(config.BINARY_MODEL_DIR, 'wb') as fout:
            pickle.dump((self.vectorizer, self.classifier), fout)
    
    def _transform_data(self, data):
        return self.vectorizer.transform(data)

    def _train(self):
        df = fm.load_dataset_as_df(config.BINARY_DATASET_DIR, column_names=['id', 'target', 'message'], usecols=['target', 'message'])

        x = df['message'].values
        y = df['target'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)

        vectorizer = CountVectorizer(strip_accents='unicode', 
                                    ngram_range=(1,2), 
                                    stop_words=nlp.get_stopwords(), 
                                    preprocessor=nlp.clean_message, 
                                    tokenizer=nlp.get_tokenizer_function(), 
                                    binary=True)

        vectorizer.fit(x_train)
        X_train = vectorizer.transform(x_train)
        X_test = vectorizer.transform(x_test)
        classifier = LogisticRegression(solver='sag', max_iter=100000)
        classifier.fit(X_train, y_train)

        fm.dump_object(config.BINARY_MODEL_DIR, classifier)
        fm.dump_object(config.BINARY_VECT_DIR, vectorizer)

        stats.get_stats_for_data(classifier, X_test, y_test)




class MLModel(Model):
    def predict(self, msg: str):
        print('Prediction of multilabel model: ' + msg)
