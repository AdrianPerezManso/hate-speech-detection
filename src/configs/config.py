import os

"""
Directories, error messages and other constants
"""

# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIRTUAL_ENV_FOLDER = '.venv/'
NLTK_DATA_FOLDER = 'nltk_data/'
MODELS_FOLDER = 'models/'
DATASETS_FOLDER = 'datasets/'
BINARY_DATASET_FOLDER = 'binary/'
MULTILABEL_DATASET_FOLDER = 'ml/'

INIT_JSON_FILENAME = 'init.json'
BINARY_MODEL_FILENAME = 'binary_classifer.pkl'
BINARY_VECT_FILENAME = 'binary_vectorizer.pkl'
MULTILABEL_MODEL_FILENAME = 'multilabel_model.pkl'
MULTILABEL_VECT_FILENAME = 'multilabel_vectorizer.pkl'
BINARY_DATASET_FILENAME = 'FinalBalancedDataset.csv'
MULTILABEL_TRAIN_DATASET_FILENAME = 'train.csv'
MULTILABEL_TEST_DATASET_FILENAME = 'test.csv'
MULTILABEL_TEST_LABELS_FILENAME = 'test_labels.csv'

INIT_JSON_DIR = os.path.join(PROJECT_ROOT, INIT_JSON_FILENAME)
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, VIRTUAL_ENV_FOLDER, NLTK_DATA_FOLDER)
BINARY_MODEL_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, BINARY_MODEL_FILENAME)
BINARY_VECT_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, BINARY_VECT_FILENAME)
MULTILABEL_MODEL_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, MULTILABEL_MODEL_FILENAME)
MULTILABEL_VECT_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, MULTILABEL_VECT_FILENAME)
BINARY_DATASET_DIR = os.path.join(DATASETS_FOLDER, BINARY_DATASET_FOLDER, BINARY_DATASET_FILENAME)
MULTILABEL_TRAIN_DATASET_DIR = os.path.join(DATASETS_FOLDER, MULTILABEL_DATASET_FOLDER, MULTILABEL_TRAIN_DATASET_FILENAME)
MULTILABEL_TEST_DATASET_DIR = os.path.join(DATASETS_FOLDER, MULTILABEL_DATASET_FOLDER, MULTILABEL_TEST_DATASET_FILENAME)
MULTILABEL_TEST_LABELS_DIR = os.path.join(DATASETS_FOLDER, MULTILABEL_DATASET_FOLDER, MULTILABEL_TEST_LABELS_FILENAME)

# init.json
ARGS = 'args'
TRAIN = 'train'
SHORT = 'short'
FLAG = 'flag'
HELP = 'help'
ACTION = 'action'
NLTK = 'nltk'
DOWNLOAD = 'download'
PACKAGES = 'packages'

# Database constants
DATABASE_FOLDER = 'database/'
DATABASE_NAME = 'users.db'
DATABASE_DIR = os.path.join(PROJECT_ROOT, DATABASE_FOLDER, DATABASE_NAME)
QUERY_GET_USER_BY_USERNAME_AND_PASSWORD = 'SELECT password FROM users WHERE username = ?'
USERNAME_FIELD = 'username'
PASSWORD_FIELD = 'password'

# Data values
NUM_TARGETS_BINARY_MODEL = 1
NUM_TARGETS_MULTILABEL_MODEL= 6
APPROPRIATE_PREDICTION = 0
INAPPROPRIATE_PREDICTION = 1
TOXIC_LABEL = 'toxic'
SEVERE_TOXIC_LABEL = 'severe_toxic'
OBSCENE_LABEL = 'obscene'
THREAT_LABEL = 'threat'
INSULT_LABEL = 'insult'
IDENTITY_HATE_LABEL = 'identity_hate'
TOXIC_LABEL_INDEX = 0
SEVERE_TOXIC_LABEL_INDEX = 1
OBSCENE_LABEL_INDEX = 2
THREAT_LABEL_INDEX = 3
INSULT_LABEL_INDEX = 4
IDENTITY_HATE_LABEL_INDEX = 5
BINARY_TARGET_VALUE = 'target'
MESSAGE_VALUE = 'message'
ID_VALUE = 'id'


# NLP
HASHTAG = '#'
LANGUAGE = 'english'
REGEX_TOKENIZER = '#\w+|&#[0-9]+;|http\S+|@?\w+'
REGEX_AT = '@'
REGEX_HTTP = 'http'
REGEX_UNICODE = '&#'
UNICODE = 'unicode'

# Max lengths
MESSAGE_MAX_LENGTH = 500
USERNAME_MAX_LENGTH = 20
PASSWORD_MAX_LENGTH = 20

# Exceptions messages
ERROR_ALREADY_AUTHENTICATED = 'You are already authenticated'
ERROR_NOT_AUTHENTICATED = 'This operation is only available for administrators'
ERROR_NOT_STRING_MESSAGE = 'Message {index} is not a valid message'
ERROR_BLANK_MESSAGE = 'Message {index} is blank'
ERROR_MAX_LENGTH_MESSAGE = 'Message {index} has exceeded maximum number of characters ({max_length})'
ERROR_FILE_WRONG_NUM_OF_COLS = 'The file does not have the correct number of columns. Discarding file'
ERROR_FILE_WRONG_EXTENSION = 'Input file has wrong extension. Discarding file'
ERROR_WRONG_NUM_OF_PREDICTION_VALUES = 'The prediction of message {index} should have {num} value/s'
ERROR_NOT_VALID_PREDICTION = 'The prediction of message {index} is not valid. Message not considered for fitting'
ERROR_NO_LAST_PREDICTION = 'Prediction not found. Try to predict a message first'
ERROR_BLANK_PREDICTION_VALUE = 'The prediction cannot be blank'
ERROR_NOT_STRING_USERNAME = 'Username not valid'
ERROR_NOT_STRING_PASSWORD = 'Password not valid'
ERROR_NOT_ALPHANUM_USERNAME = 'Username can only contain letters and numbers'
ERROR_BLANK_USERNAME = 'Username cannot be blank'
ERROR_BLANK_PASSWORD = 'Password cannot be blank'
ERROR_MAX_LENGTH_USERNAME = 'Username has exceeded maximum number of characters ({max_length})'
ERROR_MAX_LENGTH_PASSWORD = 'Password has exceeded maximum number of characters ({max_length})'
ERROR_AUTHENTICATION = 'Incorrect username or password'
ERROR_INDEX_NOT_NUMBER = 'The identification of the message has to be a number'
ERROR_NOT_VALID_INDEX = 'The message required was not found'
ERROR_INVALID_MODEL_OPT = 'Model {opt} is not valid'

# File contents
CSV_EXTENSION = '.csv'
TXT_EXTENSION = '.txt'
OUTPUT_TXT_FILE = 'The message "{msg}" has been predicted as {pred}'
OUTPUT_MESSAGE = '"{message}"'
OUTPUT_DATETIME_FORMAT = '%Y%m%d_%H%M%S'
OUTPUT_FILE_NAME = 'predictions_{datetime}{extension}'

# Stats
STATS_LABELS = ['0', '1']

