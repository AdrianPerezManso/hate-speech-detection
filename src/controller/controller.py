import numpy as np
import pandas as pd
import math
from configs import config
from classifiers.classifiers import Model, BinaryModel, MLModel
from domain.prediction import Prediction
from auth.authentication import AuthenticationModule
from utils import file_management as fm, validation

class ClassificationController:
    def __init__(self):
        self.model = BinaryModel()
        self.auth_module = AuthenticationModule()
        self.authenticated = False
        self.last_prediction = None

    def predict(self, msg: str, index: int = 0):
        result = ''
        try:
            validation.check_message_is_valid(msg, index)
            prediction = self.model.predict(np.array([msg]), index)
            result = prediction.get_predictions_for_output()
            self.last_prediction = prediction
        except Exception as e:
            print(e)
            result = config.OUTPUT_MESSAGE_NOT_PREDICTED.format(index=index + 1)
        return result

    def predict_messages(self, msgs_path: str):
        result = ''
        try:
            validation.check_file_extension(msgs_path, config.FILE_EXTENSION)
            df = fm.load_csv_as_df(msgs_path)
            validation.check_num_cols(df, 1)
            for i, message in enumerate(df.iloc[:,0].values):
                result += self.predict(message, i)
                result += ', ' if i < df.shape[0] - 1 else ''
        except Exception as e:
            print(e)
        return result


    def change_classification_method(self, model_opt: str):
       self.model = BinaryModel() if model_opt == config.OUTPUT_BINARY_MODEL else MLModel()

    def authenticate(self, usr: str, pwd: str):
        if(not self.authenticated):
            try:
                validation.check_auth_credentials_are_valid(usr, pwd)
                self.authenticated = self.auth_module.authenticate(usr, pwd)
                if(not self.authenticated): raise Exception(config.ERROR_AUTHENTICATION)
            except Exception as e:
                print(e)

    def log_off(self):
        self.authenticated = False

    def correct_predictions(self, prediction_values: list[int]):
        if(self.authenticated):
            try:
                if(not self.last_prediction): raise Exception(config.ERROR_NO_LAST_PREDICTION)
                if(not len(prediction_values)): raise Exception(config.ERROR_BLANK_PREDICTION_VALUE)
                new_pred = self.model.to_prediction(self.last_prediction._msg, 
                                                    self.last_prediction._index, 
                                                    prediction_values)
                new_pred.validate_prediction()
                self.model.fit_prediction(new_pred)
            except Exception as e:
                print(e)


    def train_models(self, model_opt: str, file_path: str):
        try:
            if(self.authenticated):
                model_to_train = self.model
                if(model_opt != self.model.get_model_opt()): model_to_train = MLModel()
                validation.check_file_extension(file_path, config.FILE_EXTENSION)
                df = fm.load_csv_as_df(file_path)
                model_to_train.fit_new_data(df)
        except Exception as e:
            print(e)

    def save_results_to_file():
        pass
    