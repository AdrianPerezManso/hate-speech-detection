from abc import ABC, abstractmethod
from joblib import load
from nlp.nlp_module import vectorize_message

class Model(ABC):
    @abstractmethod
    def predict(self, msg: str):
        pass

class BinaryModel(Model):
    def __init__(self):
        classifier = load('filename.joblib') 

    def predict(self, msg: str):
        return self.classifier.predict(vectorize_message(msg))

class MLModel(Model):
    def predict(self, msg: str):
        print('Prediction of multilabel model: ' + msg)
