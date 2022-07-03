from configs import config
from utils import validation
from abc import ABC, abstractmethod

class Prediction(ABC):
    
    def __init__(self, msg: str, index: int):
        self._msg = msg
        self._index = index

    @abstractmethod
    def get_prediction(self):
        pass

    @abstractmethod
    def get_message_for_output(self):
        pass

    @abstractmethod
    def get_predictions_for_output(self):
        pass

    @abstractmethod
    def validate_prediction(self):
        pass

    @abstractmethod
    def get_output_header(self):
        pass


class BinaryPrediction(Prediction):
    def __init__(self, msg: str, index: int,  prediction: str):
        super(BinaryPrediction, self).__init__(msg, index)
        self._prediction = prediction

    def get_prediction(self):
        return self._prediction

    def get_message_for_output(self):
        return config.MESSAGE_OUTPUT.format(message=self._msg)

    def get_predictions_for_output(self):
        prediction = config.OUTPUT_MESSAGE_APPROPRIATE if self._prediction == config.APPROPRIATE_PREDICTION else config.OUTPUT_MESSAGE_INAPPROPRIATE
        return config.BINARY_PREDICTION_FORMAT.format(index=self._index + 1, prediction=prediction)
    
    def validate_prediction(self):
        validation.check_message_is_valid(self._msg, self._index)
        validation.check_prediction_is_valid(self._prediction, self._index)
    
    def get_output_header(self):
        return [config.BINARY_MESSAGE_VALUE, config.BINARY_TARGET_VALUE]


class MLPrediction(Prediction):
    def __init__(self, msg: str, predictions: dict[str, int]):
        super(BinaryPrediction, self).__init__(msg)
        self._predictions = predictions
        self._toxic_prediction = predictions[config.TOXIC_LABEL]
        self._severe_toxic_prediction = predictions[config.SEVERE_TOXIC_LABEL]
        self._obscene_prediction = predictions[config.obscene_LABEL]
        self._threat_prediction = predictions[config.THREAT_LABEL]
        self._insult_prediction = predictions[config.INSULT_LABEL]
        self._identity_hate_prediction = predictions[config.IDENTITY_HATE_LABEL]

    def get_prediction(self):
        return self._predictions

    def get_predictions_for_output(self):
        pass

    def validate_prediction(self):
        pass
