from datetime import datetime
from configs import config, uiconfig, logconfig
from classifiers.classifiers import BinaryModel, MLModel
from domain.prediction import EmptyPrediction
from auth.authentication import AuthenticationModule
from utils import file_management as fm, validation
import logging

class ClassificationController:
    def __init__(self, train=False):
        if(train):
            logging.info(logconfig.LOG_CONTROLLER_TRAIN_START)
            BinaryModel(train)
            MLModel(train)
            logging.info(logconfig.LOG_CONTROLLER_TRAIN_END)
        self.model = BinaryModel()
        self.auth_module = AuthenticationModule()
        self.authenticated = False
        self.last_predictions = []
        logging.info(logconfig.LOG_CONTROLLER_INIT)

    def predict(self, messages: list[str]):
        logging.info(logconfig.LOG_CONTROLLER_PREDICT_START)
        result, errors = [], []
        for index, msg in enumerate(messages):
            try:
                validation.check_message_is_valid(msg, index)
                logging.info(logconfig.LOG_CONTROLLER_PREDICT_MESSAGE_VALIDATION.format(index=index+1))

                prediction = self.model.predict(msg, index)
                logging.info(logconfig.LOG_CONTROLLER_PREDICT_GET_PREDICTION.format(msg=msg, index=index+1, pred_value=prediction.get_prediction()))

                self.last_predictions.append(prediction)
                logging.debug(logconfig.LOG_CONTROLLER_PREDICT_PREDICTION_ADDED_TO_LAST_PREDICTIONS.format(index=index+1))

                result.append(prediction)
                logging.debug(logconfig.LOG_CONTROLLER_PREDICT_ADDED_TO_RESULT.format(index=index+1))
            except Exception as e:
                logging.error(logconfig.LOG_CONTROLLER_PREDICT_ERROR.format(error=e))
                
                result.append(EmptyPrediction(msg, index, str(e)))
                logging.info(logconfig.LOG_CONTROLLER_PREDICT_ADDED_INVALID_TO_RESULT.format(index=index+1))

                errors.append(str(e))
                logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_PREDICT_END)
        return result, errors

    def predict_messages_in_file(self, msgs_path: str):
        logging.info(logconfig.LOG_CONTROLLER_PREDICT_FILE_START)
        result, errors = [], []
        try:
            validation.check_file_extension(msgs_path, config.CSV_EXTENSION)
            logging.info(logconfig.LOG_CONTROLLER_PREDICT_FILE_VALIDATION_EXTENSION)

            df = fm.load_csv_as_df(msgs_path)
            logging.info(logconfig.LOG_CONTROLLER_PREDICT_FILE_LOAD_FILE.format(path=msgs_path))

            validation.check_num_cols(df, 1)
            logging.info(logconfig.LOG_CONTROLLER_PREDICT_FILE_VALIDATION_NUM_COLS.format(num=1))

            result, errors = self.predict(df.iloc[:,0].values)
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_PREDICT_FILE_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_PREDICT_FILE_END)
        return result, errors

    def redo_last_prediction(self):
        logging.info(logconfig.LOG_CONTROLLER_REDO_LAST_PREDICTION_START)
        result, errors = [], []
        try:
            if(not len(self.last_predictions)): raise Exception(config.ERROR_BLANK_PREDICTION_VALUE)
            logging.info(logconfig.LOG_CONTROLLER_LAST_PREDICTION_VALIDATION)

            last_pred_copy = self.last_predictions.copy()
            self.clear_classification()
            logging.debug(logconfig.LOG_CONTROLLER_REDO_LAST_PREDICTION_CLEAR_LAST_PREDICTION)

            result, errors = self.predict([pred._msg for pred in last_pred_copy])
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_REDO_LAST_PREDICTION_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_REDO_LAST_PREDICTION_END)
        return result, errors

    def change_classification_method(self, model_opt: str):
        logging.info(logconfig.LOG_CONTROLLER_CHANGE_METHOD_START)
        errors = []
        try:
            validation.check_correct_model_opt(model_opt)
            logging.info(logconfig.LOG_CONTROLLER_CHANGE_METHOD_VALIDATION)

            self.model = self._model_opt_to_model(model_opt)
            logging.info(logconfig.LOG_CONTROLLER_CHANGE_METHOD.format(opt=model_opt))

        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_CHANGE_METHOD_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_CHANGE_METHOD_END)
        return errors


    def authenticate(self, usr: str, pwd: str):
        logging.info(logconfig.LOG_CONTROLLER_AUTHENTICATE_START)
        errors = []
        try:
            if(not self.authenticated):
                validation.check_auth_credentials_are_valid(usr, pwd)
                logging.info(logconfig.LOG_CONTROLLER_AUTHENTICATE_VALIDATION)

                self.authenticated = self.auth_module.authenticate(usr, pwd)
                if(not self.authenticated): raise Exception(config.ERROR_AUTHENTICATION)
                logging.info(logconfig.LOG_AUTHENTICATE_SUCCESS)
            else:
                raise Exception(config.ERROR_ALREADY_AUTHENTICATED)
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_AUTHENTICATE_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_AUTHENTICATE_END.format(result=self.authenticated))
        return errors

    def correct_predictions(self, msg_index: int, prediction_values: list[int]):
        logging.info(logconfig.LOG_CONTROLLER_CORRECT_PRED_START)
        errors = []
        try:
            if(self.authenticated):
                if(not len(self.last_predictions)): raise Exception(config.ERROR_NO_LAST_PREDICTION)
                logging.info(logconfig.LOG_CONTROLLER_LAST_PREDICTION_VALIDATION)

                if(not len(prediction_values)): raise Exception(config.ERROR_BLANK_PREDICTION_VALUE)
                logging.info(logconfig.LOG_CONTROLLER_CORRECT_PRED_VALIDATION)

                last_pred = self._get_prediction_by_index(msg_index)
                new_pred = self.model.to_prediction(last_pred._msg, 
                                                    last_pred._index, 
                                                    prediction_values)
                logging.info(logconfig.LOG_CONTROLLER_CORRECT_PRED_GET_PRED)

                errors = self.model.fit_prediction(new_pred)
            else:
                raise Exception(config.ERROR_NOT_AUTHENTICATED)
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_CORRECT_PRED_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)
            
        logging.info(logconfig.LOG_CONTROLLER_CORRECT_PRED_END)
        return errors

    def train_models(self, model_opt: str, file_path: str):
        logging.info(logconfig.LOG_CONTROLLER_RETRAIN_MODEL_START)
        errors = []
        try:
            if(self.authenticated):
                model_to_train = self._model_opt_to_model(model_opt)
                if(model_to_train is None): raise Exception(config.ERROR_INVALID_MODEL_OPT)
                logging.info(logconfig.LOG_CONTROLLER_RETRAIN_MODEL_VALIDATION_MODEL_OPT.format(opt=model_to_train.get_model_opt()))

                validation.check_file_extension(file_path, config.CSV_EXTENSION)
                logging.info(logconfig.LOG_CONTROLLER_RETRAIN_MODEL_VALIDATION_FILE_EXTENSION.format(path=file_path))

                df = fm.load_csv_as_df(file_path)
                logging.info(logconfig.LOG_CONTROLLER_RETRAIN_MODEL_GET_DATA.format(path=file_path))

                errors = model_to_train.fit_new_data(df)
                self._refresh_model()
            else:
                raise Exception(config.ERROR_NOT_AUTHENTICATED)
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_RETRAIN_MODEL_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_RETRAIN_MODEL_END)
        return errors

    def save_results_to_csv(self, path):
        logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULTS_TO_CSV_START)
        errors = []
        filename = ''
        try:
            if(not len(self.last_predictions)): raise Exception(config.ERROR_NO_LAST_PREDICTION)
            logging.info(logconfig.LOG_CONTROLLER_LAST_PREDICTION_VALIDATION)

            data = [pred.construct_prediction_for_output_file() for pred in self.last_predictions]
            logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_CSV_GET_DATA)

            data_header = self.last_predictions[0].get_header_for_output_file()
            logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_CSV_GET_DATA_HEADER)

            filename = config.OUTPUT_FILE_NAME.format(datetime=datetime.now().strftime(config.OUTPUT_DATETIME_FORMAT), extension=config.CSV_EXTENSION)
            fm.create_csv_for_predictions(path, filename, data_header, data)
            logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_CSV_SUCCESS.format(path=path, file=filename))
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_CSV_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULTS_TO_CSV_END)
        return errors, filename

    def save_results_to_txt(self, path):
        logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULTS_TO_TXT_START)
        errors = []
        filename = ''
        try:
            if(not len(self.last_predictions)): raise Exception(config.ERROR_NO_LAST_PREDICTION)
            logging.info(logconfig.LOG_CONTROLLER_LAST_PREDICTION_VALIDATION)

            data = [pred.get_prediction_for_txt() for pred in self.last_predictions]
            logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_TXT_GET_DATA)

            filename = config.OUTPUT_FILE_NAME.format(datetime=datetime.now().strftime(config.OUTPUT_DATETIME_FORMAT), extension=config.TXT_EXTENSION)
            fm.create_txt_for_predictions(path, filename, data)
            logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_TXT_SUCCESS.format(path=path, file=filename))
        except Exception as e:
            logging.error(logconfig.LOG_CONTROLLER_SAVE_RESULT_TO_TXT_ERROR.format(error=e))

            errors.append(str(e))
            logging.debug(logconfig.LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS)

        logging.info(logconfig.LOG_CONTROLLER_SAVE_RESULTS_TO_TXT_END)
        return errors, filename

    def clear_classification(self):
        self.last_predictions = []
        logging.info(logconfig.LOG_CONTROLLER_CLEAR_CLASSIFICATION)

    def _model_opt_to_model(self, model_opt):
        if(model_opt == uiconfig.UI_BINARY_MODEL):
            return BinaryModel()
        elif(model_opt == uiconfig.UI_MULTILABEL_MODEL):
            return MLModel()
        else:
            return None

    def _get_prediction_by_index(self, index):
        result = list(filter(lambda pred: pred._index == index, self.last_predictions))
        if(not len(result)): raise Exception(config.ERROR_NOT_VALID_INDEX)
        logging.info(logconfig.LOG_CONTROLLER_MESSAGE_BY_INDEX_VALIDATION.format(index=index+1))

        return result[0]
    
    def _refresh_model(self):
        if(self.model.get_model_opt() == uiconfig.UI_BINARY_MODEL):
            self.change_classification_method(uiconfig.UI_MULTILABEL_MODEL)
            self.change_classification_method(uiconfig.UI_BINARY_MODEL)
        elif(self.model.get_model_opt() == uiconfig.UI_MULTILABEL_MODEL):
            self.change_classification_method(uiconfig.UI_BINARY_MODEL)
            self.change_classification_method(uiconfig.UI_MULTILABEL_MODEL)
        else:
            self.change_classification_method(uiconfig.UI_MULTILABEL_MODEL)
            self.change_classification_method(uiconfig.UI_BINARY_MODEL)
            
        logging.debug(logconfig.LOG_CONTROLLER_REFRESH_MODEL)