from configs import config
from utils import validation
from abc import ABC, abstractmethod

class Prediction(ABC):
    
    def __init__(self, msg: str, index: int):
        self._msg = msg
        self._index = index

    def get_message_for_ui(self):
        pass

    @abstractmethod
    def get_prediction_for_ui(self):
        pass

    @abstractmethod
    def validate_prediction(self):
        pass

    @abstractmethod
    def get_header_for_output_file(self):
        pass

    @abstractmethod
    def construct_prediction_for_output_file(self):
        pass


class BinaryPrediction(Prediction):
    def __init__(self, msg: str, index: int,  prediction: str):
        super(BinaryPrediction, self).__init__(msg, index)
        self._prediction = prediction

    def get_prediction(self):
        return self._prediction

    def get_message_for_ui(self):
        return config.OUTPUT_MESSAGE_PREDICTION.format(index=self._index + 1, msg=self._msg)

    def get_prediction_for_ui(self):
        prediction = config.OUTPUT_MESSAGE_APPROPRIATE if self._prediction == config.APPROPRIATE_PREDICTION else config.OUTPUT_MESSAGE_INAPPROPRIATE
        return config.OUTPUT_VALID_PREDICTION_FORMAT.format(index=self._index + 1, prediction=prediction)
    
    def validate_prediction(self):
        validation.check_message_is_valid(self._msg, self._index)
        validation.check_prediction_is_valid(self._prediction, self._index)
        return True
    
    def get_header_for_output_file(self):
        return [config.MESSAGE_VALUE, config.BINARY_TARGET_VALUE]
    
    def construct_prediction_for_output_file(self):
        return [config.MESSAGE_OUTPUT.format(message=self._msg), self._prediction]


class MLPrediction(Prediction):
    def __init__(self, msg: str, index: int, prediction: list[int]):
        super(MLPrediction, self).__init__(msg, index)
        self._prediction = prediction

    def get_prediction(self):
        return self._prediction

    def get_message_for_ui(self):
        return config.OUTPUT_MESSAGE_PREDICTION.format(index=self._index + 1, msg=self._msg)

    def get_prediction_for_ui(self):
        prediction = self._predictions_to_string()
        return config.OUTPUT_VALID_PREDICTION_FORMAT.format(index=self._index + 1, prediction=prediction)

    def validate_prediction(self):
        validation.check_message_is_valid(self._msg, self._index)
        validation.check_prediction_is_valid(self._prediction[config.TOXIC_LABEL_INDEX], self._index)
        validation.check_prediction_is_valid(self._prediction[config.SEVERE_TOXIC_LABEL_INDEX], self._index)
        validation.check_prediction_is_valid(self._prediction[config.OBSCENE_LABEL_INDEX], self._index)
        validation.check_prediction_is_valid(self._prediction[config.THREAT_LABEL_INDEX], self._index)
        validation.check_prediction_is_valid(self._prediction[config.INSULT_LABEL_INDEX], self._index)
        validation.check_prediction_is_valid(self._prediction[config.IDENTITY_HATE_LABEL_INDEX], self._index)
        return True

    def get_header_for_output_file(self):
        return [config.MESSAGE_VALUE, config.TOXIC_LABEL, config.SEVERE_TOXIC_LABEL, config.OBSCENE_LABEL,
                config.THREAT_LABEL, config.INSULT_LABEL, config.IDENTITY_HATE_LABEL]
    
    def construct_prediction_for_output_file(self):
        return [config.MESSAGE_OUTPUT.format(message=self._msg), 
                self._prediction[config.TOXIC_LABEL_INDEX], self._prediction[config.SEVERE_TOXIC_LABEL_INDEX], self._prediction[config.OBSCENE_LABEL_INDEX],
                self._prediction[config.THREAT_LABEL_INDEX], self._prediction[config.INSULT_LABEL_INDEX], self._prediction[config.IDENTITY_HATE_LABEL_INDEX]]

    def _predictions_to_string(self):
        result = '['
        result += config.OUTPUT_MESSAGE_TOXIC if self._prediction[config.TOXIC_LABEL_INDEX] == 1 else config.OUTPUT_MESSAGE_NON_TOXIC
        result += ', '
        result += config.OUTPUT_MESSAGE_SEVERE_TOXIC if self._prediction[config.SEVERE_TOXIC_LABEL_INDEX] == 1 else config.OUTPUT_MESSAGE_NON_SEVERE_TOXIC
        result += ', '
        result += config.OUTPUT_MESSAGE_OBSCENE if self._prediction[config.OBSCENE_LABEL_INDEX] == 1 else config.OUTPUT_MESSAGE_NON_OBSCENE
        result += ', '
        result += config.OUTPUT_MESSAGE_THREAT if self._prediction[config.THREAT_LABEL_INDEX] == 1 else config.OUTPUT_MESSAGE_NON_THREAT
        result += ', '
        result += config.OUTPUT_MESSAGE_INSULT if self._prediction[config.INSULT_LABEL_INDEX] == 1 else config.OUTPUT_MESSAGE_NON_INSULT
        result += ', '
        result += config.OUTPUT_MESSAGE_IDENTITY_HATE if self._prediction[config.IDENTITY_HATE_LABEL_INDEX] == 1 else config.OUTPUT_MESSAGE_NON_IDENTITY_HATE
        result += ']'
        return result

class EmptyPrediction(Prediction):
    def __init__(self, msg: str, index: int, error_msg: str):
        super(EmptyPrediction, self).__init__(msg, index)
        self._error_msg = error_msg

    def get_message_for_ui(self):
        return config.OUTPUT_MESSAGE_PREDICTION.format(index=self._index + 1, msg=self._msg)

    def get_prediction_for_ui(self):
        return config.OUTPUT_INVALID_PREDICTION_FORMAT.format(index=self._index + 1, error=self._error_msg)

    def validate_prediction(self):
        pass

    def get_header_for_output_file(self):
        pass

    def construct_prediction_for_output_file(self):
        pass