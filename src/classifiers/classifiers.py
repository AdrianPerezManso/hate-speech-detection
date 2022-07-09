import pickle
import pandas as pd
import time

from configs import config, uiconfig, logconfig
from abc import ABC, abstractmethod
from domain.prediction import Prediction, BinaryPrediction, MLPrediction
from utils import nlp, stats, file_management as fm, validation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
import logging

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
            logging.info(logconfig.LOG_CLASSIFIER_TRAIN_TRUE)
            self._train()
        else:
            logging.info(logconfig.LOG_CLASSIFIER_TRAIN_FALSE)
            with open(clf_dir, 'rb') as f:
                self.classifier = pickle.load(f)
                logging.info(logconfig.LOG_CLASSIFIER_LOAD_CLF.format(path=clf_dir))

            with open(vect_dir, 'rb') as f:
                self.vectorizer = pickle.load(f)
                logging.info(logconfig.LOG_CLASSIFIER_LOAD_VECT.format(path=vect_dir))

        logging.info(logconfig.LOG_CLASSIFIER_INIT)
    
    def predict(self, msg: str, index: int):
        logging.info(logconfig.LOG_CLASSIFIER_PREDICT_START)
        result = self.classifier.predict(self._transform_data([msg]))
        logging.info(logconfig.LOG_CLASSIFIER_PREDICT_RESULT.format(result=result))

        result_proba = self.classifier.predict_proba(self._transform_data([msg]))
        logging.info(logconfig.LOG_CLASSIFIER_PREDICT_PROBA_RESULT.format(result=result_proba))

        result = self._parse_prediction_result(result)
        logging.debug(logconfig.LOG_CLASSIFIER_PREDICT_PARSE_PREDICTION)

        return self.to_prediction(msg, index, result)

    def _partial_fit(self, x, y):
        X = self._transform_data(x)
        self.classifier.partial_fit(X, y)
    
    def _transform_data(self, data):
        return self.vectorizer.transform(data)
    
    def _filter_invalid_data(self, y: list, x: list[str], num_targets: int):
        logging.debug(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_START)
        y_filtered, x_filtered, errors = [], [], []
        for i in range(len(x)):
            try:
                msg = x[i]
                pred = y[i]
                pred_to_val = self._prepare_prediction_for_validation(pred)
                logging.debug(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_PREPARED_PRED.format(index=i))

                validation.check_prediction(msg, i, pred_to_val, num_targets)
                logging.debug(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_VALIDATION.format(index=i))

                y_filtered.append(pred)
                logging.debug(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_ADDED_TARGETS.format(index=i))

                x_filtered.append(msg)
                logging.debug(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_ADDED_FEATURES.format(index=i))
            except Exception as e:
                logging.error(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_ERROR.format(error=e))

                errors.append(str(e))
                logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.debug(logconfig.LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_END)
        return y_filtered, x_filtered, errors
    
    def _save_model(self, classifier, vectorizer):
        fm.dump_object(self.clf_dir, classifier)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_SAVE_CLF)

        fm.dump_object(self.vect_dir, vectorizer)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_SAVE_VECT)

    def _train(self):
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_START)
        start = time.time()
        df = fm.load_csv_as_df(self._get_dataset_dir(), 
                header=0, 
                column_names=self._get_headers_for_train(), 
                usecols=self._get_headers_for_train()[1:])
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_LOAD_DATASET.format(path=self._get_dataset_dir()))

        x_train, x_test, y_train, y_test = self._split_data(df)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_SPLIT_DATA)

        vectorizer = self._get_new_vectorizer()
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_GET_VECT)

        vectorizer.fit(x_train)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_VECT_FIT)

        X_train = vectorizer.transform(x_train)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_VECT_TRANSFORM_TRAIN)

        X_test = vectorizer.transform(x_test)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_VECT_TRANSFORM_TEST)
        
        classifier = self._get_new_classifier()
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_GET_CLF)

        classifier.fit(X_train, y_train)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_CLF_FIT)
        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_END)

        end = time.time()
        e_time = end - start

        logging.debug(logconfig.LOG_CLASSIFIER_TRAIN_SAVE_MODEL)
        self._save_model(classifier, vectorizer)

        logging.info(logconfig.LOG_CLASSIFIER_TRAIN_GET_STATS)
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
        logging.info(logconfig.LOG_BIN_CLASSIFIER_FIT_NEW_DATA_START)
        validation.check_num_cols(data, 2)
        logging.info(logconfig.LOG_BIN_CLASSIFIER_FIT_NEW_DATA_VALIDATION_NUM_COLS.format(num=2))

        y_new, x_new = data.iloc[:,0].values, data.iloc[:,1].values
        logging.info(logconfig.LOG_BIN_CLASSIFIER_FIT_NEW_DATA_SPLIT_DATA)

        y_new, x_new, errors = self._filter_invalid_data(y_new, x_new, num_targets=config.NUM_TARGETS_BINARY_MODEL)
        super()._partial_fit(x_new, y_new)
        logging.info(logconfig.LOG_BIN_CLASSIFIER_FIT_NEW_DATA_END)

        return errors
    
    def fit_prediction(self, prediction: BinaryPrediction):
        logging.info(logconfig.LOG_BIN_CLASSIFIER_CORRECT_PRED_START)
        df = pd.DataFrame(data={
                        0: [prediction.get_prediction()], 
                        1: [prediction._msg]
                        })
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_CORRECT_PRED_GET_DATA)

        result = self.fit_new_data(df)
        logging.info(logconfig.LOG_BIN_CLASSIFIER_CORRECT_PRED_END)

        return result

    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        validation.check_prediction(msg, index, prediction, config.NUM_TARGETS_BINARY_MODEL)
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_TO_PREDICTION_VALIDATION)

        pred = BinaryPrediction(msg, index, prediction[0])
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_PREDICT_GET_BIN_PRED)

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
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_TRAIN_GET_X)

        y = df[config.BINARY_TARGET_VALUE].values
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_TRAIN_GET_Y)

        return train_test_split(x, y, test_size=0.20, random_state=32)

    def _get_new_vectorizer(self):
        vect = CountVectorizer(strip_accents=config.UNICODE, 
                                    ngram_range=(1,2), 
                                    stop_words=nlp.get_stopwords(), 
                                    preprocessor=nlp.clean_message, 
                                    tokenizer=nlp.get_tokenizer_function(), 
                                    binary=True)
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_TRAIN_GET_VECT)

        return vect

    def _get_new_classifier(self):
        clf = SGDClassifier(loss='modified_huber', max_iter=100000)
        logging.debug(logconfig.LOG_BIN_CLASSIFIER_TRAIN_GET_CLF)

        return clf
    
    def _get_stats_for_data(self, classifier, X_test, y_test, e_time):
        stats.get_stats_for_data(classifier, X_test, y_test, e_time, make_report=True)


class MLModel(SKLearnModel):

    def __init__(self, train=False):
        super(MLModel, self).__init__(train, config.MULTILABEL_MODEL_DIR, config.MULTILABEL_VECT_DIR)

    def get_model_opt(self):
        return uiconfig.UI_MULTILABEL_MODEL
    
    def fit_new_data(self, data: pd.DataFrame):
        logging.info(logconfig.LOG_ML_CLASSIFIER_FIT_NEW_DATA_START)
        validation.check_num_cols(data, 7)
        logging.info(logconfig.LOG_ML_CLASSIFIER_FIT_NEW_DATA_VALIDATION_NUM_COLS.format(num=7))

        x_new, y_new = data.iloc[:,0].values, data.iloc[:,1:].values
        logging.info(logconfig.LOG_ML_CLASSIFIER_FIT_NEW_DATA_SPLIT_DATA)

        y_new, x_new, errors = self._filter_invalid_data(y_new, x_new, num_targets=config.NUM_TARGETS_MULTILABEL_MODEL)
        super()._partial_fit(x_new, y_new)
        logging.info(logconfig.LOG_ML_CLASSIFIER_FIT_NEW_DATA_END)

        return errors
    
    def fit_prediction(self, prediction: MLPrediction):
        logging.info(logconfig.LOG_ML_CLASSIFIER_CORRECT_PRED_START)
        df = pd.DataFrame(data={
                    0: [prediction._msg], 
                    config.TOXIC_LABEL_INDEX + 1: [prediction.get_prediction()[config.TOXIC_LABEL_INDEX]],
                    config.SEVERE_TOXIC_LABEL_INDEX + 1: [prediction.get_prediction()[config.SEVERE_TOXIC_LABEL_INDEX]],
                    config.OBSCENE_LABEL_INDEX + 1: [prediction.get_prediction()[config.OBSCENE_LABEL_INDEX]],
                    config.THREAT_LABEL_INDEX + 1: [prediction.get_prediction()[config.THREAT_LABEL_INDEX]],
                    config.INSULT_LABEL_INDEX + 1: [prediction.get_prediction()[config.INSULT_LABEL_INDEX]],
                    config.IDENTITY_HATE_LABEL_INDEX + 1: [prediction.get_prediction()[config.IDENTITY_HATE_LABEL_INDEX]]
                    })
        logging.debug(logconfig.LOG_ML_CLASSIFIER_CORRECT_PRED_GET_DATA)

        result = self.fit_new_data(df)
        logging.info(logconfig.LOG_ML_CLASSIFIER_CORRECT_PRED_END)

        return result

    def to_prediction(self, msg: str, index: int, prediction: list[int]):
        validation.check_prediction(msg, index, prediction, config.NUM_TARGETS_MULTILABEL_MODEL)
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TO_PREDICTION_VALIDATION)

        pred = MLPrediction(msg, index, prediction)
        logging.debug(logconfig.LOG_ML_CLASSIFIER_PREDICT_GET_BIN_PRED)

        return pred
    
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
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_TEST_DATASET.format(path=config.MULTILABEL_TEST_DATASET_DIR))

        df_test_labels = fm.load_csv_as_df(config.MULTILABEL_TEST_LABELS_DIR, header=0)
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_TEST_LABELS_DATASET.format(path=config.MULTILABEL_TEST_LABELS_DIR))

        df_test_labels = df_test_labels[df_test_labels[config.TOXIC_LABEL] > -1]
        df_test = df_test[df_test[config.ID_VALUE].isin(df_test_labels[config.ID_VALUE])]
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_FILTER_INVALID_ROWS)

        df_test_labels.pop(config.ID_VALUE)
        df_test.pop(config.ID_VALUE)
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_DELETE_ID_COL)
        
        df_test[[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                    config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]] = df_test_labels
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_APPEND_TEST_LABELS)

        x_train = df_train[config.MESSAGE_VALUE]
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_TRAIN_X)

        x_test = df_test[config.MESSAGE_VALUE]
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_TEST_X)

        y_train = df_train[[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                            config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]].values
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_TRAIN_Y)

        y_test = df_test[[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                          config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]].values
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_TEST_Y)
        
        return x_train, x_test, y_train, y_test

    def _get_new_vectorizer(self):
        vect = TfidfVectorizer(ngram_range=(1,1), 
                                     stop_words=nlp.get_stopwords(), 
                                     preprocessor=nlp.clean_message, 
                                     tokenizer=nlp.get_tokenizer_function(), 
                                     min_df=25)
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_VECT)

        return vect

    def _get_new_classifier(self):
        clf = MultiOutputClassifier(SGDClassifier(loss='modified_huber', max_iter=100000))
        logging.debug(logconfig.LOG_ML_CLASSIFIER_TRAIN_GET_CLF)
        
        return clf

    def _get_stats_for_data(self, classifier, X_test, y_test, e_time):
        stats.get_stats_for_data(classifier, X_test, y_test, e_time, 
                                 multilabel=True, mllabels=[config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL, 
                                                            config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL])



