import numpy as np
import pandas as pd
import math
from configs import config
from classifiers.classifiers import Model, BinaryModel, MLModel
from domain.prediction import Prediction
from auth.authentication import AuthenticationModule
from utils import file_management as fm

class Controller:
    def __init__(self):
        self.model = BinaryModel()
        self.auth_module = AuthenticationModule()
        self.authenticated = False

    def predict(self, msg: str, index: int = 1):
        result = ''
        try:
            if(not msg or pd.isnull(msg)): raise Exception(config.ERROR_NOT_STRING_MESSAGE.format(index=index))
            if(not len(msg.strip())): raise Exception(config.ERROR_BLANK_MESSAGE.format(index=index))
            if(len(msg) > config.MAX_MESSAGE_LENGTH): raise Exception(config.ERROR_MAX_LENGTH_MESSAGE.format(index=index))
            prediction = self.model.predict(np.array([msg]), index)
            result = prediction.get_predictions_for_output()
        except Exception as e:
            print(e)
            result = config.OUTPUT_MESSAGE_NOT_PREDICTED.format(index=index)
        return result

    def predict_messages(self, msgs_path: str):
        result = ''
        try:
            df = fm.load_csv_as_df(msgs_path)
            if df.shape[1] > 1: raise Exception(config.ERROR_FILE_WRONG_NUM_OF_COLS)
            for i, message in enumerate(df.iloc[:,0].values):
                result += self.predict(message, i+1)
                result += ', ' if i < df.shape[0] - 1 else ''
        except Exception as e:
            print(e)
        return result


    def change_classification_method(self, model_opt: int):
       self.model = BinaryModel() if model_opt == 0 else MLModel()

    def authenticate(self, usr: str, pwd: str):
        self.authenticated = self.auth_module.authenticate(usr, pwd)

    def log_off(self):
        self.authenticated = False

    def correct_predictions(self, predictions: list[Prediction]):
        pass

    def train_models(self, model_opt: int, file_dir: str):
        pass

    