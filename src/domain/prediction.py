from configs import config

from abc import ABC, abstractmethod

class Prediction(ABC):
    
    def __init__(self, msg: str, index: int):
        self.msg = msg
        self.index = index

    @abstractmethod
    def get_predictions(self):
        pass

    @abstractmethod
    def get_predictions_for_output(self):
        pass


class BinaryPrediction(Prediction):
    def __init__(self, msg: str, index: int,  prediction: str):
        super(BinaryPrediction, self).__init__(msg, index)
        self.prediction = prediction

    def get_predictions(self):
        return self.prediction

    def get_predictions_for_output(self):
        prediction = config.OUTPUT_MESSAGE_APPROPRIATE if self.prediction == config.APPROPRIATE_PREDICTION else config.OUTPUT_MESSAGE_INAPPROPRIATE
        return config.BINARY_PREDICTION_FORMAT.format(index=self.index, prediction=prediction)


class MLPrediction(Prediction):
    def __init__(self, msg: str, predictions: dict[str, str]):
        super(BinaryPrediction, self).__init__(msg)
        self.predictions = predictions

    def get_predictions(self):
        return self.predictions
    
    def get_predictions_for_output(self):
        pass
