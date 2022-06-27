import pickle

from abc import ABC, abstractmethod
from joblib import load
from preprocess.nlp import vectorize_messages
from train.binmodel import train_model

class Model(ABC):
    @abstractmethod
    def predict(self, msg: str):
        pass

class BinaryModel(Model):
    def __init__(self):
        # train_model()
        self.classifier = load('models/binary_classifier.joblib') 
        self.vectorizer = pickle.load(open('models/vectorizer.pickle', 'rb'))

    def predict(self, msg: str):
        return self.classifier.predict(vectorize_messages(msg))[0]

class MLModel(Model):
    def predict(self, msg: str):
        print('Prediction of multilabel model: ' + msg)
