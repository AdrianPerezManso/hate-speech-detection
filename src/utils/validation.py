import pandas as pd
from configs import config

def check_message_is_string(msg: str):
    return msg and not pd.isnull(msg)

def check_message_is_not_blank(msg: str):
    return len(msg.strip())

def check_message_not_max_len(msg: str, max_length: int):
    return len(msg) <= max_length

def check_prediction_is_valid(pred: int, index: int):
    if (pred != config.APPROPRIATE_PREDICTION and 
        pred != config.INAPPROPRIATE_PREDICTION):
            raise Exception(config.ERROR_NOT_VALID_PREDICTION.format(index=index + 1))
    return True

def check_num_cols(df: pd.DataFrame, expected_num_col: int):
    if(df.shape[1] != expected_num_col): 
        raise Exception(config.ERROR_FILE_WRONG_NUM_OF_COLS)

def check_message_is_valid(msg: str, index: int):
    if(not check_message_is_string(msg)): raise Exception(config.ERROR_NOT_STRING_MESSAGE.format(index=index + 1))
    if(not check_message_is_not_blank(msg)): raise Exception(config.ERROR_BLANK_MESSAGE.format(index=index + 1))
    if(not check_message_not_max_len(msg, config.MAX_MESSAGE_LENGTH)): 
        raise Exception(config.ERROR_MAX_LENGTH_MESSAGE.format(index=index + 1).format(max_length=config.MESSAGE_MAX_LENGTH))
    return True

def check_auth_credentials_are_valid(usr: str, pwd: str):
    if(not check_message_is_string(usr)): raise Exception(config.ERROR_NOT_STRING_USERNAME)
    if(not check_message_is_string(pwd)): raise Exception(config.ERROR_NOT_STRING_PASSWORD)
    if(not check_message_is_not_blank(usr)): raise Exception(config.ERROR_BLANK_USERNAME)
    if(not check_message_is_not_blank(pwd)): raise Exception(config.ERROR_BLANK_PASSWORD)
    if(not check_message_not_max_len(usr, config.USERNAME_MAX_LENGTH)): raise Exception(config.ERROR_MAX_LENGTH_USERNAME.format(max_length=config.USERNAME_MAX_LENGTH))
    if(not check_message_not_max_len(pwd, config.PASSWORD_MAX_LENGTH)): raise Exception(config.ERROR_MAX_LENGTH_PASSWORD.format(max_length=config.PASSWORD_MAX_LENGTH))
    return True

def check_file_extension(path: str, extension: str):
    if(not path.endswith(extension)): raise Exception(config.ERROR_FILE_WRONG_EXTENSION)