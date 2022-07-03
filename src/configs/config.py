import os

# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_FOLDER = 'models/'
DATASETS_FOLDER = 'datasets/'
BINARY_DATASET_FOLDER = 'binary/'
MULTILABEL_DATASET_FOLDER = 'ml/'
OUTPUT_FOLDER = 'out/'

BINARY_MODEL_FILENAME = 'binary_classifer.pkl'
BINARY_VECT_FILENAME = 'binary_vectorizer.pkl'
MULTILABEL_MODEL_FILENAME = 'multilabel.pkl'
BINARY_DATASET_FILENAME = 'FinalBalancedDataset.csv'
MULTILABEL_DATASET_FILENAME = ''

BINARY_MODEL_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, BINARY_MODEL_FILENAME)
BINARY_VECT_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, BINARY_VECT_FILENAME)
MULTILABEL_MODEL_DIR = os.path.join(PROJECT_ROOT, MODELS_FOLDER, MULTILABEL_MODEL_FILENAME)
BINARY_DATASET_DIR = os.path.join(DATASETS_FOLDER, BINARY_DATASET_FOLDER, BINARY_DATASET_FILENAME)
MULTILABEL_DATASET_DIR = os.path.join(DATASETS_FOLDER, MULTILABEL_DATASET_FOLDER, MULTILABEL_DATASET_FILENAME)

OUTPUT_FILE_DIR = os.path.join(PROJECT_ROOT, OUTPUT_FOLDER)
DATETIME_OUTPUT_FORMAT = '%Y%m%d_%H%M%S'
OUTPUT_FILE_NAME = 'predictions_{datetime}{extension}' 

# Database constants
DATABASE_NAME = 'users.db'
QUERY_GET_USER_BY_USERNAME_AND_PASSWORD = 'SELECT password FROM users WHERE username = ?'
USERNAME_FIELD = 'username'
PASSWORD_FIELD = 'password'

# UI Messages constants
OUTPUT_BINARY_MODEL = 'Binary'
OUTPUT_MULTILABEL_MODEL = 'Itemized'

OUTPUT_MESSAGE_APPROPRIATE = 'Appropriate'
OUTPUT_MESSAGE_INAPPROPRIATE = 'Inappropriate'
BINARY_PREDICTION_FORMAT = 'Message: {index}, Predicted: {prediction}'

OUTPUT_MESSAGE_TOXIC = 'Toxic'
OUTPUT_MESSAGE_NON_TOXIC = 'Not toxic'
OUTPUT_MESSAGE_SEVERE_TOXIC = 'Severe toxic'
OUTPUT_MESSAGE_NON_SEVERE_TOXIC = 'Not severe toxic'
OUTPUT_MESSAGE_OBSCENE = 'Obscene'
OUTPUT_MESSAGE_NON_OBSCENE = 'Not obscene'
OUTPUT_MESSAGE_THREAT = 'Threat'
OUTPUT_MESSAGE_NON_THREAT = 'Not threat'
OUTPUT_MESSAGE_INSULT = 'Insult'
OUTPUT_MESSAGE_NON_INSULT = 'Not insult'
OUTPUT_MESSAGE_IDENTITY_HATE = 'Identity hate'
OUTPUT_MESSAGE_NON_IDENTITY_HATE = 'Not identity hate'

OUTPUT_MESSAGE_NOT_PREDICTED = 'Message: {index}, Not predicted'

# Data values
APPROPRIATE_PREDICTION = 0
INAPPROPRIATE_PREDICTION = 1
TOXIC_LABEL = 'toxic'
SEVERE_TOXIC_LABEL = 'severe_toxic'
OBSCENE_LABEL = 'obscene'
THREAT_LABEL = 'threat'
INSULT_LABEL = 'insult'
IDENTITY_HATE_LABEL = 'identity_hate'
BINARY_TARGET_VALUE = 'target'
BINARY_MESSAGE_VALUE = 'message'

# NLP
LANGUAGE = 'english'
REGEX_TOKENIZER = '#\w+|&#[0-9]+;|http\S+|@?\w+'
AT = '@'
HTTP = 'http'
UNICODE = '&#'

# Max lengths
MESSAGE_MAX_LENGTH = 500
USERNAME_MAX_LENGTH = 20
PASSWORD_MAX_LENGTH = 20

# Exceptions messages
ERROR_NOT_STRING_MESSAGE = 'Message {index} is not a valid message'
ERROR_BLANK_MESSAGE = 'Message {index} is blank'
ERROR_MAX_LENGTH_MESSAGE = 'Message {index} has exceeded maximum number of characters ({max_length})'
ERROR_FILE_WRONG_NUM_OF_COLS = 'The file does not have the correct number of columns. Discarding file'
ERROR_FILE_WRONG_EXTENSION = 'Input file has wrong extension. Discarding file'
ERROR_NOT_VALID_PREDICTION = 'The prediction of message {index} is not valid. Message not considered for fitting'
ERROR_NO_LAST_PREDICTION = 'Prediction not found. Try to predict a message first'
ERROR_BLANK_PREDICTION_VALUE = 'The prediction cannot be blank'
ERROR_NOT_STRING_USERNAME = 'Username not valid'
ERROR_NOT_STRING_PASSWORD = 'Password not valid'
ERROR_BLANK_USERNAME = 'Username cannot be blank'
ERROR_BLANK_PASSWORD = 'Password cannot be blank'
ERROR_MAX_LENGTH_USERNAME = 'Username has exceeded maximum number of characters ({max_length})'
ERROR_MAX_LENGTH_PASSWORD = 'Password has exceeded maximum number of characters ({max_length})'
ERROR_AUTHENTICATION = 'Incorrect username or password'

# File contents
FILE_EXTENSION = '.csv'
MESSAGE_OUTPUT = '"{message}"'

