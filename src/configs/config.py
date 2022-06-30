import os

# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_FOLDER = 'models/'
DATASETS_FOLDER = 'datasets/'
BINARY_DATASET_FOLDER = 'binary/'
MULTILABEL_DATASET_FOLDER = 'ml/'

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

# Database constants
DATABASE_NAME = 'users.db'
QUERY_GET_USER_BY_USERNAME_AND_PASSWORD = 'SELECT password FROM users WHERE username = ?'
USERNAME_FIELD = 'username'
PASSWORD_FIELD = 'password'

# UI Messages constants
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

# Predictions
APPROPRIATE_PREDICTION = 0
INAPPROPRIATE_PREDICTION = 1

# NLP
LANGUAGE = 'english'
REGEX_TOKENIZER = '#\w+|&#[0-9]+;|http\S+|@?\w+'
AT = '@'
HTTP = 'http'
UNICODE = '&#'

# Message configuration
MAX_MESSAGE_LENGTH = 500

# Exceptions messages
ERROR_NOT_STRING_MESSAGE = 'Message {index} is not a valid message'
ERROR_BLANK_MESSAGE = 'Message {index} is blank. Not predicted'
ERROR_MAX_LENGTH_MESSAGE = 'Message {index} has exceeded maximum characters. Not predicted'
ERROR_FILE_WRONG_NUM_OF_COLS = 'The file does not have the correct number of columns. Discarding file'
