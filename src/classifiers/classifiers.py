import pickle
import pandas as pd
import time

from configs import config
from abc import ABC, abstractmethod
from joblib import load
from domain.prediction import Prediction, BinaryPrediction, MLPrediction
from utils import nlp, stats, file_management as fm, validation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.memory_profiler import MemoryProfiler 

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

class BinaryModel(Model):
    def __init__(self, train=False):
        self.vectorizer = None
        self.classifier = None
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
        result = self.classifier.predict(self._transform_data([msg]))[0]
        #print(self.classifier.predict_proba(self._transform_data([msg])))
        return BinaryPrediction(msg, index, result)
    
    def fit_new_data(self, data: pd.DataFrame):
        validation.check_num_cols(data, 2)
        y_new, x_new = data.iloc[:,0].values, data.iloc[:,1].values
        y_new, x_new, errors = self._filter_invalid_data(y_new, x_new)
        X_new = self._transform_data(x_new)
        self.classifier.partial_fit(X_new, y_new)
        return errors
    
    def fit_prediction(self, prediction: BinaryPrediction):
        df = pd.DataFrame(data={
                        0: [prediction.get_prediction()], 
                        1: [prediction._msg]
                        })
        return self.fit_new_data(df)

    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        validation.check_prediction(msg, index, prediction, config.NUM_TARGETS_BINARY_MODEL)
        pred = BinaryPrediction(msg, index, prediction[0])
        return pred
    
    def _filter_invalid_data(self, y: list[int], x: list[str]):
        y_filtered, x_filtered, errors = [], [], []
        for i in range(len(x)):
            try:
                msg = x[i]
                pred = y[i] 
                validation.check_prediction(msg, i, [pred], config.NUM_TARGETS_BINARY_MODEL)
                y_filtered.append(pred)
                x_filtered.append(msg)
            except Exception as e:
                print(e)
                errors.append(str(e))
        return y_filtered, x_filtered, errors

    def _transform_data(self, data):
        return self.vectorizer.transform(data)

    def _save_model(self, classifier, vectorizer):
        fm.dump_object(config.BINARY_MODEL_DIR, classifier)
        fm.dump_object(config.BINARY_VECT_DIR, vectorizer)

    def _train(self):
        start = time.time()
        df = fm.load_csv_as_df(config.BINARY_DATASET_DIR, 
                header=0, 
                column_names=[config.ID_VALUE, config.BINARY_TARGET_VALUE, config.MESSAGE_VALUE], 
                usecols=[config.BINARY_TARGET_VALUE, config.MESSAGE_VALUE])

        x = df[config.MESSAGE_VALUE].values
        y = df[config.BINARY_TARGET_VALUE].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)


        vectorizer = CountVectorizer(strip_accents=config.UNICODE, 
                                    ngram_range=(1,2), 
                                    stop_words=nlp.get_stopwords(), 
                                    preprocessor=nlp.clean_message, 
                                    tokenizer=nlp.get_tokenizer_function(), 
                                    binary=True)
        
        vectorizer.fit(x_train)
        X_train = vectorizer.transform(x_train)
        X_test = vectorizer.transform(x_test)
        
        classifier = SGDClassifier(loss='modified_huber', max_iter=100000)
        classifier.fit(X_train, y_train)
        end = time.time()
        e_time = end - start
        self._save_model(classifier, vectorizer)
        stats.get_stats_for_data(classifier, X_test, y_test, e_time=e_time, make_report=True)




class MLModel(Model):
    def __init__(self, train=False):
        self.vectorizer = None
        self.classifier = None
        if(train):
            self._train()
        else:
            with open(config.MULTILABEL_VECT_DIR, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(config.MULTILABEL_MODEL_DIR, 'rb') as f:
                self.classifier = pickle.load(f)

    def get_model_opt(self):
        return config.OUTPUT_MULTILABEL_MODEL
    
    def predict(self, msg: str, index: int):
        prediction = self.classifier.predict(self._transform_data([msg]))[0]
        #print(self.classifier.predict_proba(self._transform_data([msg])))
        return MLPrediction(msg, index, prediction)
    
    def fit_new_data(self, data: pd.DataFrame):
        print(data)
        validation.check_num_cols(data, 7)
        x_new, y_new = data.iloc[:,0].values, data.iloc[:,1:].values
        x_new, y_new, errors = self._filter_invalid_data(y_new, x_new)
        X_new = self._transform_data(x_new)
        self.classifier.partial_fit(X_new, y_new)
        return errors
    
    def fit_prediction(self, prediction: MLPrediction):
        df = pd.DataFrame(data={
                    0: [prediction._msg], 
                    config.TOXIC_LABEL_INDEX + 1: [prediction.get_prediction()[config.TOXIC_LABEL_INDEX]],
                    config.SEVERE_TOXIC_LABEL_INDEX + 1: [prediction.get_prediction()[config.SEVERE_TOXIC_LABEL_INDEX]],
                    config.OBSCENE_LABEL_INDEX + 1: [prediction.get_prediction()[config.OBSCENE_LABEL_INDEX]],
                    config.THREAT_LABEL_INDEX + 1: [prediction.get_prediction()[config.THREAT_LABEL_INDEX]],
                    config.INSULT_LABEL_INDEX + 1: [prediction.get_prediction()[config.INSULT_LABEL_INDEX]],
                    config.IDENTITY_HATE_LABEL_INDEX + 1: [prediction.get_prediction()[config.IDENTITY_HATE_LABEL_INDEX]]
                    })
        return self.fit_new_data(df)
    
    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        validation.check_prediction(msg, index, prediction, config.NUM_TARGETS_MULTILABEL_MODEL)
        return MLPrediction(msg, index, prediction)
    
    def _transform_data(self, data):
        return self.vectorizer.transform(data)
    
    def _filter_invalid_data(self, y: list[list[int]], x: list[str]):
        x_filtered, y_filtered, errors = [], [], []
        for i in range(len(x)):
            try:
                msg = x[i]
                pred = y[i] 
                pred_values = [pred[config.TOXIC_LABEL_INDEX], pred[config.SEVERE_TOXIC_LABEL_INDEX],
                               pred[config.OBSCENE_LABEL_INDEX], pred[config.THREAT_LABEL_INDEX],
                               pred[config.INSULT_LABEL_INDEX], pred[config.IDENTITY_HATE_LABEL_INDEX]]
                validation.check_prediction(msg, i, pred_values, config.NUM_TARGETS_MULTILABEL_MODEL)
                y_filtered.append(pred)
                x_filtered.append(msg)
            except Exception as e:
                print(e)
                errors.append(str(e))
        return x_filtered, y_filtered, errors

    def _train(self):
        start = time.time()
        df_train = fm.load_csv_as_df(config.MULTILABEL_TRAIN_DATASET_DIR, 
                header=0, 
                column_names=[config.ID_VALUE, config.MESSAGE_VALUE, 
                              config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                              config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL], 
                usecols=[config.MESSAGE_VALUE, 
                         config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                         config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL])
            
        df_test = fm.load_csv_as_df(config.MULTILABEL_TEST_DATASET_DIR, header=0, column_names=[config.ID_VALUE, config.MESSAGE_VALUE])
        df_test_labels = fm.load_csv_as_df(config.MULTILABEL_TEST_LABELS_DIR, header=0)
        df_test_labels = df_test_labels[df_test_labels[config.TOXIC_LABEL] > -1]
        df_test = df_test[df_test[config.ID_VALUE].isin(df_test_labels[config.ID_VALUE])]
        df_test_labels.pop(config.ID_VALUE)
        df_test.pop(config.ID_VALUE)
        
        df_test[[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                    config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]] = df_test_labels 

        x_train = df_train[config.MESSAGE_VALUE]
        x_test = df_test[config.MESSAGE_VALUE]
        y_train = df_train[[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                            config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]].values
        y_test = df_test[[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                          config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]].values
        
        vectorizer = TfidfVectorizer(ngram_range=(1,1), 
                                     stop_words=nlp.get_stopwords(), 
                                     preprocessor=nlp.clean_message, 
                                     tokenizer=nlp.get_tokenizer_function(), 
                                     min_df=25)

        vectorizer.fit(x_train)
        X_train = vectorizer.transform(x_train)
        X_test = vectorizer.transform(x_test)
        
        classifier = MultiOutputClassifier(SGDClassifier(loss='modified_huber', max_iter=100000))
        classifier.fit(X_train, y_train)
        end = time.time()
        e_time = end - start
        self._save_model(classifier, vectorizer)
        stats.get_stats_for_data(classifier, X_test, y_test, e_time=e_time, multilabel=True, 
                                mllabels=[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL, 
                                        config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL])

    def _save_model(self, classifier, vectorizer):
        fm.dump_object(config.MULTILABEL_MODEL_DIR, classifier)
        fm.dump_object(config.MULTILABEL_VECT_DIR, vectorizer)

