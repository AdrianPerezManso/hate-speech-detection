import pickle
import pandas as pd
import os

from configs import config
from abc import ABC, abstractmethod
from joblib import load
from domain.prediction import Prediction, BinaryPrediction, MLPrediction
from utils import nlp, stats, file_management as fm, validation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    @abstractmethod
    def get_model_opt(self):
        pass
    @abstractmethod
    def predict(self, msg: str, index: int):
        pass
    @abstractmethod
    def fit_new_data(self, data):
        pass
    @abstractmethod
    def fit_prediction(self, prediction: Prediction):
        pass
    @abstractmethod
    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        pass
    @abstractmethod
    def _transform_data(self, data):
        pass

class BinaryModel(Model):
    def __init__(self, train=False):
        if(train): 
            self._train()
        else:
            with open(config.BINARY_VECT_DIR, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(config.BINARY_MODEL_DIR, 'rb') as f:
                self.classifier = pickle.load(f)
    
    def get_model_opt(self):
        return config.OUTPUT_BINARY_MODEL

    def predict(self, msg: str, index: int):
        result = self.classifier.predict(self._transform_data(msg))[0]
        return BinaryPrediction(msg[0], index, result)
    
    def fit_new_data(self, data: pd.DataFrame):
        validation.check_num_cols(data, 2)
        y_new, x_new = data.iloc[:,0].values, data.iloc[:,1].values
        y_new, x_new = self._filter_invalid_data(y_new, x_new)
        df_new = pd.DataFrame(data={
                        config.BINARY_TARGET_VALUE: [int(y) for y in y_new], 
                        config.BINARY_MESSAGE_VALUE: x_new
                        })
        df_origin = fm.load_csv_as_df(config.BINARY_DATASET_DIR,
                        header=0, 
                        usecols=[1,2], 
                        column_names=[config.BINARY_TARGET_VALUE, config.BINARY_MESSAGE_VALUE])
        df = pd.concat([df_origin, df_new])
        self._train(df)
    
    def fit_prediction(self, prediction: BinaryPrediction):
        df = pd.DataFrame(data={
                        0: [prediction.get_prediction()], 
                        1: [prediction._msg]
                        })
        self.fit_new_data(df)

    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        return BinaryPrediction(msg, index, prediction[0])
    
    def _filter_invalid_data(self, y: list[int], x: list[str]):
        y_filtered, x_filtered = [], []
        for i in range(len(x)):
            try:
                msg = x[i]
                pred = y[i] 
                validation.check_message_is_valid(msg, i)
                validation.check_prediction_is_valid(pred, i)
                y_filtered.append(pred)
                x_filtered.append(msg)
            except Exception as e:
                print(e)
        return y_filtered, x_filtered

    def _transform_data(self, data):
        return self.vectorizer.transform(data)

    def _save_model(self):
        fm.dump_object(config.BINARY_MODEL_DIR, self.classifier)
        fm.dump_object(config.BINARY_VECT_DIR, self.vectorizer)

    def _train(self, df_new):
        if(df_new is not None):
            df = df_new
        else:
            df = fm.load_csv_as_df(config.BINARY_DATASET_DIR, 
                    header=0, 
                    column_names=['id', 'target', 'message'], 
                    usecols=['target', 'message'])

        x = df['message'].values
        y = df['target'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)

        if(self.vectorizer):
            vectorizer = self.vectorizer
        else:
            vectorizer = CountVectorizer(strip_accents='unicode', 
                                        ngram_range=(1,2), 
                                        stop_words=nlp.get_stopwords(), 
                                        preprocessor=nlp.clean_message, 
                                        tokenizer=nlp.get_tokenizer_function(), 
                                        binary=True)

        vectorizer.fit(x_train)
        X_train = vectorizer.transform(x_train)
        X_test = vectorizer.transform(x_test)
        
        if(self.classifier):
            classifier = self.classifier
        else:
            classifier = LogisticRegression(solver='sag', max_iter=100000)

        classifier.fit(X_train, y_train)
        self._save_model()
        stats.get_stats_for_data(classifier, X_test, y_test)




class MLModel(Model):
    def predict(self, msg: str, index: int):
        print('Prediction of multilabel model: ' + msg)
