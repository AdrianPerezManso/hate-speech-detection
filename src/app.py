import numpy as np
from models import Model, BinaryModel, MLModel

class App:
    def __init__(self):
        self.model = BinaryModel()

    def predict(self, msg: str):
        prediction = self.model.predict(np.array([msg]))
        return 'apropiado' if prediction == '0' else 'inapropiado'

    def setModel(self, model: Model):
        self.model = model