import pickle
import pandas as pd
import time

from configs import config, uiconfig
from abc import ABC, abstractmethod
from domain.prediction import Prediction, BinaryPrediction, MLPrediction
from utils import nlp, stats, file_management as fm, validation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier

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

class SKLearnModel(Model, ABC):

    def __init__(self, train, clf_dir, vect_dir):
        self.vectorizer = None
        self.classifier = None
        self.clf_dir = clf_dir
        self.vect_dir = vect_dir
        if(train):
            self._train()
        else:
            with open(clf_dir, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(vect_dir, 'rb') as f:
                self.vectorizer = pickle.load(f)
    
    def predict(self, msg: str, index: int):
        result = self.classifier.predict(self._transform_data([msg]))
        #print(self.classifier.predict_proba(self._transform_data([msg])))
        result = self._parse_prediction_result(result)
        return self.to_prediction(msg, index, result)

    def _partial_fit(self, x, y):
        X = self._transform_data(x)
        self.classifier.partial_fit(X, y)
    
    def _transform_data(self, data):
        return self.vectorizer.transform(data)
    
    def _filter_invalid_data(self, y: list, x: list[str], num_targets: int):
        y_filtered, x_filtered, errors = [], [], []
        for i in range(len(x)):
            try:
                msg = x[i]
                pred = y[i]
                pred_to_val = self._prepare_prediction_for_validation(pred) 
                validation.check_prediction(msg, i, pred_to_val, num_targets)
                y_filtered.append(pred)
                x_filtered.append(msg)
            except Exception as e:
                print(e)
                errors.append(str(e))
        return y_filtered, x_filtered, errors
    
    def _save_model(self, classifier, vectorizer):
        fm.dump_object(self.clf_dir, classifier)
        fm.dump_object(self.vect_dir, vectorizer)

    def _train(self):
        start = time.time()
        df = fm.load_csv_as_df(self._get_dataset_dir(), 
                header=0, 
                column_names=self._get_headers_for_train(), 
                usecols=self._get_headers_for_train()[1:])

        x_train, x_test, y_train, y_test = self._split_data(df)


        vectorizer = self._get_new_vectorizer()
        
        vectorizer.fit(x_train)
        X_train = vectorizer.transform(x_train)
        X_test = vectorizer.transform(x_test)
        
        classifier = self._get_new_classifier()
        classifier.fit(X_train, y_train)
        end = time.time()
        e_time = end - start
        self._save_model(classifier, vectorizer)
        self._get_stats_for_data(classifier, X_test, y_test, e_time)

    @abstractmethod        
    def get_model_opt(self):
        pass

    @abstractmethod
    def fit_new_data(self):
        pass

    @abstractmethod
    def to_prediction(msg: str, index: int, result):
        pass
    
    @abstractmethod
    def fit_prediction(self, prediction: Prediction):
        pass

    @abstractmethod
    def _parse_prediction_result(self, result):
        pass

    @abstractmethod
    def _prepare_prediction_for_validation(self, prediction):
        pass
    
    @abstractmethod
    def _get_dataset_dir(self):
        pass

    @abstractmethod
    def _get_headers_for_train(self):
        pass
    
    @abstractmethod
    def _split_data(self, df):
        pass

    @abstractmethod
    def _get_new_vectorizer(self):
        pass

    @abstractmethod
    def _get_new_classifier(self):
        pass
    
    @abstractmethod
    def _get_stats_for_data(self, classifier, X_test, y_test, e_time):
        pass


class BinaryModel(SKLearnModel):

    def __init__(self, train=False):
        super(BinaryModel, self).__init__(train, config.BINARY_MODEL_DIR, config.BINARY_VECT_DIR)

    def get_model_opt(self):
        return uiconfig.UI_BINARY_MODEL
    
    def fit_new_data(self, data: pd.DataFrame):
        validation.check_num_cols(data, 2)
        y_new, x_new = data.iloc[:,0].values, data.iloc[:,1].values
        y_new, x_new, errors = self._filter_invalid_data(y_new, x_new, num_targets=config.NUM_TARGETS_BINARY_MODEL)
        super()._partial_fit(x_new, y_new)
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
    
    def _parse_prediction_result(self, result):
        return result
    
    def _prepare_prediction_for_validation(self, prediction):
        return [prediction]
    
    def _get_dataset_dir(self):
        return config.BINARY_DATASET_DIR

    def _get_headers_for_train(self):
        return [config.ID_VALUE, config.BINARY_TARGET_VALUE, config.MESSAGE_VALUE]
    
    def _split_data(self, df):
        x = df[config.MESSAGE_VALUE].values
        y = df[config.BINARY_TARGET_VALUE].values
        return train_test_split(x, y, test_size=0.20, random_state=32)

    def _get_new_vectorizer(self):
        return CountVectorizer(strip_accents=config.UNICODE, 
                                    ngram_range=(1,2), 
                                    stop_words=nlp.get_stopwords(), 
                                    preprocessor=nlp.clean_message, 
                                    tokenizer=nlp.get_tokenizer_function(), 
                                    binary=True)

    def _get_new_classifier(self):
        return SGDClassifier(loss='modified_huber', max_iter=100000)
    
    def _get_stats_for_data(self, classifier, X_test, y_test, e_time):
        stats.get_stats_for_data(classifier, X_test, y_test, e_time, make_report=True)


class MLModel(SKLearnModel):

    def __init__(self, train=False):
        super(MLModel, self).__init__(train, config.MULTILABEL_MODEL_DIR, config.MULTILABEL_VECT_DIR)

    def get_model_opt(self):
        return uiconfig.UI_MULTILABEL_MODEL
    
    def fit_new_data(self, data: pd.DataFrame):
        validation.check_num_cols(data, 7)
        x_new, y_new = data.iloc[:,0].values, data.iloc[:,1:].values
        y_new, x_new, errors = self._filter_invalid_data(y_new, x_new, num_targets=config.NUM_TARGETS_MULTILABEL_MODEL)
        super()._partial_fit(x_new, y_new)
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
        print(df)
        return self.fit_new_data(df)

    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        validation.check_prediction(msg, index, prediction, config.NUM_TARGETS_MULTILABEL_MODEL)
        return MLPrediction(msg, index, prediction)
    
    def _parse_prediction_result(self, result):
        return result[0]
    
    def _prepare_prediction_for_validation(self, prediction):
        return prediction
    
    def _get_dataset_dir(self):
        return config.MULTILABEL_TRAIN_DATASET_DIR

    def _get_headers_for_train(self):
        return [config.ID_VALUE, config.MESSAGE_VALUE, 
                config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]
    
    def _split_data(self, df_train):
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
        
        return x_train, x_test, y_train, y_test

    def _get_new_vectorizer(self):
        return TfidfVectorizer(ngram_range=(1,1), 
                                     stop_words=nlp.get_stopwords(), 
                                     preprocessor=nlp.clean_message, 
                                     tokenizer=nlp.get_tokenizer_function(), 
                                     min_df=25)

    def _get_new_classifier(self):
        return MultiOutputClassifier(SGDClassifier(loss='modified_huber', max_iter=100000))

    def _get_stats_for_data(self, classifier, X_test, y_test, e_time):
        stats.get_stats_for_data(classifier, X_test, y_test, e_time, 
                                 multilabel=True, mllabels=[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL, 
                                                            config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL])



