from abc import ABC, abstractmethod

class Prediction(ABC):
    
    def __init__(self, msg: str):
        self.msg = msg

    def get_message(self):
        return self.msg

    @abstractmethod
    def get_predictions(self):
        pass


class BinaryPrediction(Prediction):
    def __init__(self, msg: str, prediction: str):
        super(BinaryPrediction, self).__init__(msg)
        self.prediction = prediction

    def get_predictions(self):
        return self.prediction


class MLPrediction(Prediction):
    def __init__(self, msg: str, predictions: dict[str, str]):
        super(BinaryPrediction, self).__init__(msg)
        self.predictions = predictions

    def get_predictions(self):
        return self.predictions
