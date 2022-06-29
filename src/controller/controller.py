import numpy as np
from configs import config
from classifiers.classifiers import Model, BinaryModel, MLModel
from domain.prediction import Prediction
from auth.authentication import AuthenticationModule

class Controller:
    def __init__(self):
        self.model = BinaryModel()
        self.auth_module = AuthenticationModule()
        self.authenticated = False

    def predict(self, msg: str):
        if(msg.strip() == ''): return ''
        if(len(msg) > config.MAX_MESSAGE_LENGTH): return ''
        prediction = self.model.predict(np.array([msg]))
        return prediction.get_predictions_for_output()

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

    