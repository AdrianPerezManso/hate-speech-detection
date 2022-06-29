from configs import config

from abc import ABC, abstractmethod

class Prediction(ABC):
    
    def __init__(self, msg: str):
        self.msg = msg

    def get_message(self):
        return self.msg

    @abstractmethod
    def get_predictions(self):
        pass

    @abstractmethod
    def get_predictions_for_output(self):
        pass


class BinaryPrediction(Prediction):
    def __init__(self, msg: str, prediction: str):
        super(BinaryPrediction, self).__init__(msg)
        self.prediction = prediction

    def get_predictions(self):
        return self.prediction

    def get_predictions_for_output(self):
        print(type(self.prediction))
        print(self.prediction == config.APPROPRIATE_PREDICTION)
        return config.OUTPUT_MESSAGE_APPROPRIATE if self.prediction == config.APPROPRIATE_PREDICTION else config.OUTPUT_MESSAGE_INAPPROPRIATE


class MLPrediction(Prediction):
    def __init__(self, msg: str, predictions: dict[str, str]):
        super(BinaryPrediction, self).__init__(msg)
        self.predictions = predictions

    def get_predictions(self):
        return self.predictions
    
    def get_predictions_for_output(self):
        pass
