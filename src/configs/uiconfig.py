import os

"""
Messages, labels and text values of the user interface
"""

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELP_FOLDER = 'help/'

HELP_SUBMIT_MESSAGES_FILENAME = 'submit_messages.txt'
HELP_TRAIN_DATA_FILENAME = 'train_data.txt'

HELP_SUBMIT_MESSAGES_DIR = os.path.join(PROJECT_ROOT, HELP_FOLDER, HELP_SUBMIT_MESSAGES_FILENAME)
HELP_TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, HELP_FOLDER, HELP_TRAIN_DATA_FILENAME)

# Messages on the UI
UI_BINARY_MODEL = 'Binary'
UI_MULTILABEL_MODEL = 'Itemized'
UI_MESSAGE_APPROPRIATE = 'Appropriate'
UI_MESSAGE_INAPPROPRIATE = 'Inappropriate'
UI_VALID_PREDICTION_FORMAT = 'Message {index}'
UI_INVALID_PREDICTION_FORMAT = 'No prediction. Reason: {error}'
UI_MESSAGE_TOXIC = 'Toxic'
UI_MESSAGE_NON_TOXIC = 'Not toxic'
UI_MESSAGE_SEVERE_TOXIC = 'Severe toxic'
UI_MESSAGE_NON_SEVERE_TOXIC = 'Not severe toxic'
UI_MESSAGE_OBSCENE = 'Obscene'
UI_MESSAGE_NON_OBSCENE = 'Not obscene'
UI_MESSAGE_THREAT = 'Threat'
UI_MESSAGE_NON_THREAT = 'Not threat'
UI_MESSAGE_INSULT = 'Insult'
UI_MESSAGE_NON_INSULT = 'Not insult'
UI_MESSAGE_IDENTITY_HATE = 'Identity hate'
UI_MESSAGE_NON_IDENTITY_HATE = 'Not identity hate'
UI_MESSAGE_NOT_PREDICTED = 'Message: {index}, Not predicted'
UI_MESSAGE_PREDICTION = '{index}: {msg}'
UI_MESSAGE_CSV_FILE_DETECTED = 'File .csv detected. Press Classify'
UI_MESSAGE_SAVED_FILE = 'Saved {file}'
UI_MESSAGE_PRED_CORRECTED_SUCCESS = 'Prediction successfully corrected'
UI_MESSAGE_CONFIRMATION_CORRECT_PREDICTION = 'Operation:\t Predict messages\nMessage {index}:\t "{msg}"\nOld prediction:\t {old_value}\nNew prediction:\t {new_value}\n'
UI_MESSAGE_CONFIRMATION_TRAIN_MODEL = 'Operation:\t Train model\nModel to train:\t {model_opt} model\nData:\t {file}'
UI_MESSAGE_TRAINING_SUCCESS = 'Training was performed successfully'
UI_MESSAGE_HELP_BIN_TRAIN_FILE_FORMAT = 'Example for a row in the file:\n0,”he was a boy”'
UI_MESSAGE_HELP_ML_TRAIN_FILE_FORMAT = 'Example for a row in the file:\n“this is a message”,0,0,0,0,0,0'
UI_MESSAGE_PERFORMING_OPERATION = 'Performing operation. This might take some time, please wait'

# Buttons, labels, combos, texts, text areas, text inputs texts values
UI_LABEL_MSG_TXT_AREA = 'Write a message:'
UI_LABEL_PRED_TXT_AREA = 'Prediction:'
UI_CLASSIFY_BTN = 'Classify'
UI_CLEAR_ALL_BTN = 'Clear All'
UI_REFRESH_BTN = 'Refresh'
UI_SUBMIT_MESSAGES_FILE_BTN = 'Submit messages file'
UI_SUBMIT_MESSAGES_TXT = '...Submitted data.csv'
UI_HELP_BTN = 'ⓘ'
UI_SUBMIT_TRAIN_FILE_BTN = '⚙'
UI_LABEL_METHOD_COMBO = 'Select classification method:'
UI_AUTHENTICATE_BTN = 'Log as administrator'
UI_TRAIN_MODEL_BTN = 'Train model'
UI_SAVE_CSV_BTN = 'Save results to .csv file'
UI_SAVE_TXT_BTN = 'Save results to .txt file'
UI_LABEL_N_MESSAGE_COMBO = 'Nº of the message:'
UI_CORRECT_PRED_BTN = 'Correct'
UI_CANCEL_BTN = 'Cancel'
UI_LABEL_CORRECT_PRED_BIN_COMBO = 'Inappropriate:'
UI_LABEL_CORRECT_PRED = 'Correct predictions'
UI_LABEL_USR_IN = 'Username:'
UI_LABEL_PWD_IN = 'Password:'
UI_SUBMIT_BTN = 'Submit'
UI_LABEL_SUBMIT_TRAIN_FILE = 'Upload file:'
UI_LABEL_CONFIRM_BTN = 'Proceed with operation?'
UI_CONFIRM_BTN = 'Confirm'
UI_OK_BTN = 'OK'

#Element keys
UI_KEY_EXIT = 'Exit'
UI_KEY_MSG_TXT_AREA = '<msg_txt_area>'
UI_KEY_PRED_TXT_AREA = '<pred_txt_area>'
UI_KEY_METHOD_COMBO = '<method_combo>'
UI_KEY_MSG_FILE_BTN = '<msg_file_btn>'
UI_KEY_MSG_FILE_TXT = '<msg_file_txt>'
UI_KEY_CLASSIFY_BTN = '<classify_btn>'
UI_KEY_REFRESH_CLASSIFY_BTN = '<refresh_classify_btn>'
UI_KEY_CLEAR_BTN = '<clear_btn>'
UI_KEY_SAVE_CSV_BTN = '<save_csv_btn>'
UI_KEY_SAVE_TXT_BTN = '<save_txt_btn>'
UI_KEY_AUTH_BTN = '<auth_btn>'
UI_KEY_TRAIN_BTN = '<train_btn>'
UI_KEY_N_MSG_COMBO = '<n_msg_combo>'
UI_KEY_SUBMIT_CORRECT_PRED_BTN = '<submit_correct_pred_btn>'
UI_KEY_CANCEL_CORRECT_PRED_BTN = '<cancel_correct_pred_btn>'
UI_KEY_BIN_CORRECT_PRED_COMBO = '<bin_correct_pred_combo>'
UI_KEY_TOXIC_CORRECT_PRED_COMBO = '<toxic_correct_pred_combo>'
UI_KEY_SEVERE_TOXIC_CORRECT_PRED_COMBO = '<severe_toxic_correct_pred_combo>'
UI_KEY_OBSCENE_CORRECT_PRED_COMBO = '<obscene_correct_pred_combo>'
UI_KEY_THREAT_CORRECT_PRED_COMBO = '<threat_correct_pred_combo>'
UI_KEY_INSULT_CORRECT_PRED_COMBO = '<insult_correct_pred_combo>'
UI_KEY_IDENTITY_HATE_CORRECT_PRED_COMBO = '<identity_hate_correct_pred_combo>'
UI_KEY_HELP_BTN = '<help_btn>'
UI_KEY_CORRECT_PRED_BIN_PANEL = '<correct_pred_panel_binary>'
UI_KEY_FIRST_CORRECT_PRED_ML_PANEL = '<first_correct_pred_panel_ml>'
UI_KEY_SECOND_CORRECT_PRED_ML_PANEL = '<second_correct_pred_panel_ml>'
UI_KEY_THIRD_CORRECT_PRED_ML_PANEL = '<third_correct_pred_panel_ml>'
UI_KEY_CORRECT_PRED_PANEL = '<correct_pred_panel>'
UI_KEY_SUBMIT_BTN = '<submit_btn>'
UI_KEY_USR_IN = '<usr_in>'
UI_KEY_PWD_IN ='<pwd_in>'
UI_KEY_MSG_TXT = '<msg_txt>'
UI_KEY_CANCEL_BTN = '<cancel_btn>'
UI_KEY_FINE_IN = '<file_in>'
UI_KEY_FILE_BTN = '<file_btn>'
UI_KEY_EXAMPLE_TXT = '<example_txt>'
UI_KEY_OK_BTN = '<ok_btn>'
UI_KEY_CONFIRM_BTN = '<confirm_btn>'
UI_KEY_CONFIRM_TXT = '<confirm_txt>'

# Window titles
UI_TITLE_ERROR = 'ERROR'
UI_TITLE_ERROR_SAVE_RESULT = 'ERROR: Save result'
UI_TITLE_SAVED_FILE = 'Saved file'
UI_TITLE_CORRECT_PREDICTIONS = 'Correcting predictions'
UI_TITLE_MAIN_WINDOW = 'Detect inappropriate messages'
UI_TITLE_AUTHENTICATION_WINDOW = 'Log as Administrator'
UI_TITLE_TRAINING_CONFIRMATION_WINDOW = 'Training model'
UI_TITLE_TRAIN_MODEL_WINDOW = 'Train'
UI_TITLE_HELP_WINDOW = 'Help'

UI_PASSWORD_CHARACTER = '*'