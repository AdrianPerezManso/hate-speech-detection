import numpy as np
from datetime import datetime
from configs import config, uiconfig
from classifiers.classifiers import Model, BinaryModel, MLModel
from domain.prediction import Prediction, EmptyPrediction
from auth.authentication import AuthenticationModule
from utils import file_management as fm, validation

class ClassificationController:
    def __init__(self, train=False):
        if(train):
            BinaryModel(train)
            MLModel(train)
        self.model = BinaryModel()
        self.auth_module = AuthenticationModule()
        self.authenticated = False
        self.last_predictions = []

    def predict(self, messages: list[str]):
        result, errors = [], []
        for index, msg in enumerate(messages):
            try:
                validation.check_message_is_valid(msg, index)
                prediction = self.model.predict(msg, index)
                self.last_predictions.append(prediction)
                result.append(prediction)
            except Exception as e:
                print(e)
                result.append(EmptyPrediction(msg, index, str(e)))
                errors.append(str(e))
        return result, errors

    def predict_messages_in_file(self, msgs_path: str):
        result, errors = [], []
        try:
            validation.check_file_extension(msgs_path, config.CSV_EXTENSION)
            df = fm.load_csv_as_df(msgs_path)
            validation.check_num_cols(df, 1)
            result, errors = self.predict(df.iloc[:,0].values)
        except Exception as e:
            print(e)
            errors.append(str(e))
        return result, errors

    def redo_last_prediction(self):
        result, errors = [], []
        try:
            if(not len(self.last_predictions)): raise Exception(config.ERROR_BLANK_PREDICTION_VALUE)
            last_pred_copy = self.last_predictions.copy()
            self.clear_classification()
            result, errors = self.predict([pred._msg for pred in last_pred_copy])
        except Exception as e:
            print(e)
            errors.append(str(e))
        return result, errors

    def change_classification_method(self, model_opt: str):
        errors = []
        try:
            validation.check_correct_model_opt(model_opt)
            self.model = self._model_opt_to_model(model_opt)
        except Exception as e:
            print(e)
            errors.append(str(e))
        return errors


    def authenticate(self, usr: str, pwd: str):
        errors = []
        try:
            if(not self.authenticated):
                validation.check_auth_credentials_are_valid(usr, pwd)
                self.authenticated = self.auth_module.authenticate(usr, pwd)
                if(not self.authenticated): raise Exception(config.ERROR_AUTHENTICATION)
            else:
                raise Exception(config.ERROR_ALREADY_AUTHENTICATED)
        except Exception as e:
            print(e)
            errors.append(str(e))
        return errors

    def correct_predictions(self, msg_index: int, prediction_values: list[int]):
        errors = []
        try:
            if(self.authenticated):
                if(not len(self.last_predictions)): raise Exception(config.ERROR_NO_LAST_PREDICTION)
                if(not len(prediction_values)): raise Exception(config.ERROR_BLANK_PREDICTION_VALUE)
                last_pred = self._get_prediction_by_index(msg_index)
                new_pred = self.model.to_prediction(last_pred._msg, 
                                                    last_pred._index, 
                                                    prediction_values)
                
                errors = self.model.fit_prediction(new_pred)
            else:
                raise Exception(config.ERROR_NOT_AUTHENTICATED)
        except Exception as e:
            print(e)
            errors.append(str(e))
        return errors

    def train_models(self, model_opt: str, file_path: str):
        errors = []
        try:
            if(self.authenticated):
                model_to_train = self._model_opt_to_model(model_opt)
                validation.check_file_extension(file_path, config.CSV_EXTENSION)
                df = fm.load_csv_as_df(file_path)
                errors = model_to_train.fit_new_data(df)
                self._refresh_model()
            else:
                raise Exception(config.ERROR_NOT_AUTHENTICATED)
        except Exception as e:
            print(e)
            errors.append(str(e))
        return errors

    def save_results_to_csv(self, path):
        errors = []
        filename = ''
        try:
            if(not len(self.last_predictions)): raise Exception(config.ERROR_NO_LAST_PREDICTION)
            data = [pred.construct_prediction_for_output_file() for pred in self.last_predictions]
            data_header = self.last_predictions[0].get_header_for_output_file()
            filename = config.OUTPUT_FILE_NAME.format(datetime=datetime.now().strftime(config.OUTPUT_DATETIME_FORMAT), extension=config.CSV_EXTENSION)
            fm.create_csv_for_predictions(path, filename, data_header, data)
        except Exception as e:
            errors.append(str(e))
            print(e)
        return errors, filename

    def save_results_to_txt(self, path):
        errors = []
        filename = ''
        try:
            if(not len(self.last_predictions)): raise Exception(config.ERROR_NO_LAST_PREDICTION)
            data = [pred.get_prediction_for_txt() for pred in self.last_predictions]
            filename = config.OUTPUT_FILE_NAME.format(datetime=datetime.now().strftime(config.OUTPUT_DATETIME_FORMAT), extension=config.TXT_EXTENSION)
            fm.create_txt_for_predictions(path, filename, data)
        except Exception as e:
            errors.append(str(e))
            print(e)
        return errors, filename

    def clear_classification(self):
        self.last_predictions = []

    def _model_opt_to_model(self, model_opt):
        if(model_opt == uiconfig.UI_BINARY_MODEL):
            return BinaryModel()
        elif(model_opt == uiconfig.UI_MULTILABEL_MODEL):
            return MLModel()

    def _get_prediction_by_index(self, index):
        result = list(filter(lambda pred: pred._index == index, self.last_predictions))
        if(not len(result)): raise Exception(config.ERROR_NOT_VALID_INDEX)
        return result[0]
    
    def _refresh_model(self):
        if(self.model.get_model_opt() == uiconfig.UI_BINARY_MODEL):
            self.change_classification_method(uiconfig.UI_MULTILABEL_MODEL)
            self.change_classification_method(uiconfig.UI_BINARY_MODEL)
        elif(self.model.get_model_opt() == uiconfig.UI_MULTILABEL_MODEL):
            self.change_classification_method(uiconfig.UI_BINARY_MODEL)
            self.change_classification_method(uiconfig.UI_MULTILABEL_MODEL)
        else:
            self.change_classification_method(uiconfig.UI_MULTILABEL_MODEL)
            self.change_classification_method(uiconfig.UI_BINARY_MODEL)