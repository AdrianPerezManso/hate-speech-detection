from configs import config
from datetime import datetime
import os
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = 'log/'
OUTPUT_LOG_FILE_DIR = os.path.join(PROJECT_ROOT, LOG_FOLDER)

OUTPUT_LOG_FILENAME = 'session' + datetime.now().strftime(config.OUTPUT_DATETIME_FORMAT) + '.log'

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [module: %(module)s] [function: %(funcName)s()] [line %(lineno)s] - %(message)s'
LOG_DATE_FORMAT = '%d-%b-%y %H:%M:%S'

LOG_MAIN_START_OF_APPLICATION = 'Booting program'
LOG_MAIN_END_OF_APPLICATION = 'Closed window. End of program'
LOG_MAIN_FLAG_PARSING = 'Parsed flags... -t flag was {t}'

LOG_CONTROLLER_INIT = 'Initialized controller'
LOG_CONTROLLER_TRAIN_START = 'Training models'
LOG_CONTROLLER_TRAIN_END = 'Sucessful training'
LOG_CONTROLLER_PREDICT_START = 'Started prediction process'
LOG_CONTROLLER_PREDICT_MESSAGE_VALIDATION = 'Validated message {index}'
LOG_CONTROLLER_PREDICT_GET_PREDICTION = 'Obtained prediction for Message {index} "{msg}": {pred_value}'
LOG_CONTROLLER_PREDICT_PREDICTION_ADDED_TO_LAST_PREDICTIONS = 'Message {index} added to last_predictions'
LOG_CONTROLLER_PREDICT_ADDED_TO_RESULT = 'Prediction of message {index} added to result'
LOG_CONTROLLER_PREDICT_ERROR = 'ERROR during prediction: {error}'
LOG_CONTROLLER_PREDICT_ADDED_INVALID_TO_RESULT = 'INVALID Prediction of message {index} added to result'
LOG_CONTROLLER_ADDED_ERROR_TO_ERRORS = 'Added error to errors'
LOG_CONTROLLER_PREDICT_END = 'Finalized prediction process'
LOG_CONTROLLER_PREDICT_FILE_START = 'Started prediction of file process'
LOG_CONTROLLER_PREDICT_FILE_VALIDATION_EXTENSION = 'Validated file extension'
LOG_CONTROLLER_PREDICT_FILE_ERROR = 'ERROR during prediction of file messages: {error}'
LOG_CONTROLLER_PREDICT_FILE_LOAD_FILE = 'File {path} sucessfully loaded'
LOG_CONTROLLER_PREDICT_FILE_VALIDATION_NUM_COLS = 'File has correct number of columns ({num})'
LOG_CONTROLLER_PREDICT_FILE_END = 'Finalized prediction of file process'
LOG_CONTROLLER_REDO_LAST_PREDICTION_START = 'Started redo last prediction process'
LOG_CONTROLLER_LAST_PREDICTION_VALIDATION = 'Validated existence of last predictions'
LOG_CONTROLLER_REDO_LAST_PREDICTION_CLEAR_LAST_PREDICTION = 'Removed last predictions'
LOG_CONTROLLER_REDO_LAST_PREDICTION_ERROR = 'ERROR during redo prediction process: {error}'
LOG_CONTROLLER_REDO_LAST_PREDICTION_END = 'Finalized redo last prediction process'
LOG_CONTROLLER_CHANGE_METHOD_START = 'Started classification method process'
LOG_CONTROLLER_CHANGE_METHOD_END = 'Finalized classification method process'
LOG_CONTROLLER_CHANGE_METHOD_VALIDATION = 'Validated method option'
LOG_CONTROLLER_CHANGE_METHOD = 'Changed classification method to "{opt}"'
LOG_CONTROLLER_CHANGE_METHOD_ERROR = 'ERROR during change of classification method: {error}'
LOG_CONTROLLER_AUTHENTICATE_START = 'Started authentication process'
LOG_CONTROLLER_AUTHENTICATE_VALIDATION = 'Valid username and password'
LOG_CONTROLLER_AUTHENTICATE_ERROR = 'ERROR during authentication: {error}'
LOG_CONTROLLER_AUTHENTICATE_END = 'Finalized authentication process. Access granted: {result}'
LOG_CONTROLLER_CORRECT_PRED_START = 'Started correct prediction process'
LOG_CONTROLLER_CORRECT_PRED_VALIDATION = 'Validated existence of new predictions'
LOG_CONTROLLER_MESSAGE_BY_INDEX_VALIDATION = 'Found message with index {index}'
LOG_CONTROLLER_CORRECT_PRED_GET_PRED = 'Obtained Prediction object'
LOG_CONTROLLER_CORRECT_PRED_ERROR = 'ERROR during the correction of prediction: {error}'
LOG_CONTROLLER_CORRECT_PRED_END = 'Finalized correct prediction process'
LOG_CONTROLLER_RETRAIN_MODEL_START = 'Started retrain model process'
LOG_CONTROLLER_RETRAIN_MODEL_VALIDATION_MODEL_OPT = 'Validated model option "{opt}"'
LOG_CONTROLLER_RETRAIN_MODEL_VALIDATION_FILE_EXTENSION = 'Validated extension of file {path}'
LOG_CONTROLLER_RETRAIN_MODEL_GET_DATA = 'Data successfully loaded from {path}'
LOG_CONTROLLER_RETRAIN_MODEL_ERROR = 'ERROR during mode retraining: {error}'
LOG_CONTROLLER_RETRAIN_MODEL_END = 'Finalized retrain model process'
LOG_CONTROLLER_REFRESH_MODEL = 'Refreshed model'
LOG_CONTROLLER_SAVE_RESULTS_TO_CSV_START = 'Started save results to .csv file process'
LOG_CONTROLLER_SAVE_RESULT_TO_CSV_GET_DATA = 'Obtained data from Prediction object'
LOG_CONTROLLER_SAVE_RESULT_TO_CSV_GET_DATA_HEADER = 'Obtained data header from Prediction object'
LOG_CONTROLLER_SAVE_RESULT_TO_CSV_SUCCESS = 'Sucessfully created {file} in {path}'
LOG_CONTROLLER_SAVE_RESULT_TO_CSV_ERROR = 'ERROR during save of .csv file: {error}'
LOG_CONTROLLER_SAVE_RESULTS_TO_CSV_END = 'Finalized save results to .csv file process'
LOG_CONTROLLER_SAVE_RESULTS_TO_TXT_START = 'Started save results to .txt file process'
LOG_CONTROLLER_SAVE_RESULT_TO_TXT_GET_DATA = 'Obtained data from Prediction object'
LOG_CONTROLLER_SAVE_RESULT_TO_TXT_SUCCESS = 'Sucessfully created {file} in {path}'
LOG_CONTROLLER_SAVE_RESULT_TO_TXT_ERROR = 'ERROR during save of .txt file: {error}'
LOG_CONTROLLER_SAVE_RESULTS_TO_TXT_END = 'Finalized save results to .txt file process'
LOG_CONTROLLER_CLEAR_CLASSIFICATION = 'Removed all last predictions'

LOG_AUTHENTICATE_INIT = 'Initialized authentication module'
LOG_AUTHENTICATE_EXECUTED_QUERY = 'Query executed'
LOG_AUTHENTICATE_FAIL = 'The authentication was not sucessful. No administrator access granted'
LOG_AUTHENTICATE_SUCCESS = 'The authentication was sucessful. Administrator access granted'

LOG_BIN_PREDICTION_INIT = 'Initialized BinaryPrediction with Message {index} "{msg}" and prediction {pred}'
LOG_ML_PREDICTION_INIT = 'Initialized MLPrediction with Message {index} "{msg}" and prediction {pred}'

LOG_USER_REPO_INIT = 'Initialized user repository'
LOG_USER_REPO_CREATED_CON = 'Established connection with database'
LOG_USER_REPO_CLOSED_CON = 'Closed connection with database'
LOG_USER_REPO_ERROR = 'ERROR in user repository: {error}'

LOG_NLP_START = 'Text preprocessing process started'
LOG_NLP_LOWER = 'Converted text to lowercase'
LOG_NLP_TOKEN = 'Tokenized test'
LOG_NLP_HASHTAG_PROCESSING = 'Found hashtag. Split hashtag into words'
LOG_NLP_REMOVE_USERNAME_LINK_UNICODE = 'Removed usernames, links and unicode symbols'
LOG_NLP_STEM = 'Obtained text stems'
LOG_NLP_END = 'Finalized text preprocessing process'

LOG_CLASSIFIER_LOAD_CLF = 'Loaded classifier from {path}'
LOG_CLASSIFIER_LOAD_VECT = 'Loaded vectorizer from {path}'
LOG_CLASSIFIER_INIT = 'Initialized model'
LOG_CLASSIFIER_TRAIN_TRUE = 'Train is True. Training models'
LOG_CLASSIFIER_TRAIN_FALSE = 'Train is False. Obtaining created model'
LOG_CLASSIFIER_TRAIN_START = 'Starting training'
LOG_CLASSIFIER_TRAIN_LOAD_DATASET = 'Dataset successfully loaded from {path}'
LOG_CLASSIFIER_TRAIN_SPLIT_DATA = 'Data split into features and targets for training and tests sets'
LOG_CLASSIFIER_TRAIN_GET_VECT = 'Obtained vectorizer'
LOG_CLASSIFIER_TRAIN_VECT_FIT = 'Training features fit into vectorizer'
LOG_CLASSIFIER_TRAIN_VECT_TRANSFORM_TRAIN = 'Training features transformed by vectorizer'
LOG_CLASSIFIER_TRAIN_VECT_TRANSFORM_TEST = 'Test features transformed by vectorizer'
LOG_CLASSIFIER_TRAIN_GET_CLF = 'Obtained classifier'
LOG_CLASSIFIER_TRAIN_CLF_FIT = 'Training fit into classifier'
LOG_CLASSIFIER_TRAIN_END = 'Finalized training'
LOG_CLASSIFIER_TRAIN_SAVE_MODEL = 'Saving vectorizer and classifier'
LOG_CLASSIFIER_TRAIN_SAVE_CLF = 'Saved classifier'
LOG_CLASSIFIER_TRAIN_SAVE_VECT = 'Saved vectorizer'
LOG_CLASSIFIER_TRAIN_GET_STATS = 'Getting validation statistics'
LOG_CLASSIFIER_PREDICT_START = 'Started prediction process'
LOG_CLASSIFIER_PREDICT_RESULT = 'Obtained prediction result: {result}'
LOG_CLASSIFIER_PREDICT_PROBA_RESULT = 'Obtained probabilities of result: {result}'
LOG_CLASSIFIER_PREDICT_PARSE_PREDICTION = 'Parsed prediction for Prediction class'
LOG_CLASSIFIER_PREDICT_END = 'Finalized prediction process'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_START = 'Started filtering invalid data process'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_PREPARED_PRED = 'Prepared Message {index} prediction for validation'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_VALIDATION = 'Validated Message {index}'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_ADDED_TARGETS = 'Message {index} targets added to result'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_ADDED_FEATURES = 'Message {index} features added to result'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_ERROR = 'ERROR during retraining: {error}'
LOG_CLASSIFIER_FIT_NEW_DATA_FILTER_INVALID_DATA_END = 'Finalized filtering invalid data process'

LOG_BIN_CLASSIFIER_TRAIN_GET_X = 'Obtained features from data'
LOG_BIN_CLASSIFIER_TRAIN_GET_Y = 'Obtained targets from data'
LOG_BIN_CLASSIFIER_TRAIN_GET_VECT = 'Obtained CountVectorizer'
LOG_BIN_CLASSIFIER_TRAIN_GET_CLF = 'Obtained SGDClassifier'
LOG_BIN_CLASSIFIER_TO_PREDICTION_VALIDATION = 'Validated message and prediction value'
LOG_BIN_CLASSIFIER_PREDICT_GET_BIN_PRED = 'Created BinaryPrediction object'
LOG_BIN_CLASSIFIER_CORRECT_PRED_START = 'Started correction of binary prediction process'
LOG_BIN_CLASSIFIER_CORRECT_PRED_GET_DATA = 'Built data for binary model'
LOG_BIN_CLASSIFIER_CORRECT_PRED_END = 'Finalized correction of binary prediction process'
LOG_BIN_CLASSIFIER_FIT_NEW_DATA_START = 'Started retraining binary model process'
LOG_BIN_CLASSIFIER_FIT_NEW_DATA_VALIDATION_NUM_COLS = 'Data has correct number of columns for binary model({num})'
LOG_BIN_CLASSIFIER_FIT_NEW_DATA_SPLIT_DATA = 'Data split into features and target'
LOG_BIN_CLASSIFIER_FIT_NEW_DATA_END = 'Finalized retraining binary model process'

LOG_ML_CLASSIFIER_TRAIN_GET_TEST_DATASET = 'Multilabel validation dataset successfully loaded from {path}'
LOG_ML_CLASSIFIER_TRAIN_GET_TEST_LABELS_DATASET = 'Multilabel validation labels dataset successfully loaded from {path}'
LOG_ML_CLASSIFIER_TRAIN_FILTER_INVALID_ROWS = 'Removed samples with -1 predictions'
LOG_ML_CLASSIFIER_TRAIN_DELETE_ID_COL = 'Removed id column for validation and validation labels datasets'
LOG_ML_CLASSIFIER_TRAIN_APPEND_TEST_LABELS = 'Appended labels to validation dataset'
LOG_ML_CLASSIFIER_TRAIN_GET_TRAIN_X = 'Obtained features from training data'
LOG_ML_CLASSIFIER_TRAIN_GET_TEST_X = 'Obtained features from validation data'
LOG_ML_CLASSIFIER_TRAIN_GET_TRAIN_Y = 'Obtained targets from training data'
LOG_ML_CLASSIFIER_TRAIN_GET_TEST_Y = 'Obtained targets from validation data'
LOG_ML_CLASSIFIER_TRAIN_GET_VECT = 'Obtained TfidfVectorizer'
LOG_ML_CLASSIFIER_TRAIN_GET_CLF = 'Obtained MultiOutput SGDClassifier'
LOG_ML_CLASSIFIER_TO_PREDICTION_VALIDATION = 'Validated message and prediction values'
LOG_ML_CLASSIFIER_PREDICT_GET_BIN_PRED = 'Created MLPrediction object'
LOG_ML_CLASSIFIER_CORRECT_PRED_START = 'Started correction of multilabel prediction process'
LOG_ML_CLASSIFIER_CORRECT_PRED_GET_DATA = 'Built data for multilabel model'
LOG_ML_CLASSIFIER_CORRECT_PRED_END = 'Finalized correction of multilabel prediction process'
LOG_ML_CLASSIFIER_FIT_NEW_DATA_START = 'Started retraining multilabel model process'
LOG_ML_CLASSIFIER_FIT_NEW_DATA_VALIDATION_NUM_COLS = 'Data has correct number of columns for multilabel model({num})'
LOG_ML_CLASSIFIER_FIT_NEW_DATA_SPLIT_DATA = 'Data split into features and targets'
LOG_ML_CLASSIFIER_FIT_NEW_DATA_END = 'Finalized retraining multilabel model process'


LOG_STATS_TRAIN_PREDICTION = 'Validation test prediction completed'
LOG_STATS_METRIC_ACCURACY = 'Accuracy: {result}'
LOG_STATS_METRIC_PRECISION = 'Precision: {result}'
LOG_STATS_METRIC_RECALL = 'Recall: {result}'
LOG_STATS_METRIC_F1 = 'F1 Score: {result}'
LOG_STATS_METRIC_HAMMING = 'Hamming loss: {result}'
LOG_STATS_METRIC_JACCARD = 'Jaccard score: {result}'
LOG_STATS_METRIC_EXEC_TIME = 'Execution time: {result}'