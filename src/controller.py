import numpy as np
from models import Model, BinaryModel, MLModel
from prediction import Prediction
from authentication import AuthenticationModule

class Controller:
    def __init__(self):
        self.model = BinaryModel()
        self.auth_module = AuthenticationModule()
        self.authenticated = False

    def predict(self, msg: str):
        prediction = self.model.predict(np.array([msg]))
        return 'apropiado' if prediction == '0' else 'inapropiado'

    def predict(self, msgs: list[str]):
        pass

    def change_classification_method(self, model_opt: int):
       pass

    def authenticate(self, usr: str, pwd: str):
        self.authenticated = self.auth_module.authenticate(usr, pwd)

    def log_off(self):
        self.authenticated = False

    def correct_predictions(self, predictions: list[Prediction]):
        pass

    def train_models(self, model_opt: int, file_dir: str):
        pass

    