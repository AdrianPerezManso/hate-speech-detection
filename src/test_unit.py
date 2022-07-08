from controller.controller import ClassificationController
from auth.authentication import AuthenticationModule 
from configs import config
from classifiers.classifiers import BinaryModel, MLModel
from domain.prediction import EmptyPrediction
import unittest
import time


class BusinessTestClass(unittest.TestCase):

    def test_correct_initialization(self):
        """Initialization test"""
        controller = ClassificationController()
        self.assertTrue(controller is not None)
        self.assertTrue(isinstance(controller.auth_module, AuthenticationModule))
        self.assertFalse(controller.authenticated)
        self.assertEqual(len(controller.last_predictions), 0)

    def test_1_1(self):
        controller = ClassificationController()
        errors = controller.change_classification_method('Binary')
        self.assertEqual(len(errors), 0)
        self.assertTrue(isinstance(controller.model, BinaryModel))

        errors = controller.change_classification_method('Itemized')
        self.assertEqual(len(errors), 0)
        self.assertTrue(isinstance(controller.model, MLModel))

    def test_1_2(self):
        controller = ClassificationController()
        errors = controller.change_classification_method('wrong model')
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_INVALID_MODEL_OPT)

    def test_2_1(self):
        controller = ClassificationController()
        result, errors = controller.predict(['', ' '])
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0], config.ERROR_NOT_STRING_MESSAGE.format(index=result[0]._index + 1))
        self.assertEqual(errors[1], config.ERROR_BLANK_MESSAGE.format(index=result[1]._index + 1))

    def test_2_2(self):
        controller = ClassificationController()
        valid_message = 'a'*499
        valid_message_2 = 'a'*500
        too_long_message = 'a'*501

        result, errors = controller.predict([valid_message, valid_message_2, too_long_message])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[-1], config.ERROR_MAX_LENGTH_MESSAGE.format(index=result[-1]._index + 1, max_length = 500))
        self.assertTrue(isinstance(result[-1], EmptyPrediction))

    def test_2_3(self):
        controller = ClassificationController()
        txt_path = config.PROJECT_ROOT + '/testfiles/' + 'wrong_extension.txt'
        _, errors = controller.predict_messages_in_file(txt_path)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_FILE_WRONG_EXTENSION)

    def test_2_4(self):
        controller = ClassificationController()
        path = config.PROJECT_ROOT + '/testfiles/' + 'file_with_bad_messages.csv'
        result, errors = controller.predict_messages_in_file(path)
        self.assertEqual(len(errors), 2)
        self.assertEqual(len(result), 7)
        self.assertEqual(errors[0], config.ERROR_MAX_LENGTH_MESSAGE.format(index=result[1]._index + 1, max_length = 500))
        self.assertEqual(errors[1], config.ERROR_NOT_STRING_MESSAGE.format(index=result[3]._index + 1))

    def test_2_7(self):
        controller = ClassificationController()      
        valid_message = 'hi there'
        result, errors = controller.predict([valid_message])
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]._prediction is not None)
    
    def test_2_8(self):
        controller = ClassificationController()
        path = config.PROJECT_ROOT + '/testfiles/' + 'messages.csv'
        result, errors = controller.predict_messages_in_file(path)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(result), 3)
        self.assertTrue(result[0]._prediction is not None)
        self.assertTrue(result[1]._prediction is not None)
        self.assertTrue(result[2]._prediction is not None)

    def test_3_1(self):
        controller = ClassificationController()
        errors, _ = controller.save_results_to_csv()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NO_LAST_PREDICTION)

    def test_3_2(self):
        controller = ClassificationController()
        message = 'hi there'
        _, _ = controller.predict([message])
        errors, filename = controller.save_results_to_csv()
        self.assertEqual(len(errors), 0)
        self.assertTrue(filename != '')
    
    def test_4_1(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        errors = controller.correct_predictions(0, [1])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NO_LAST_PREDICTION)

    def test_4_2(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        _, _ = controller.predict(['hi there'])
        errors = controller.correct_predictions(0, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_BLANK_PREDICTION_VALUE)
        controller.clear_classification()

        result, _ = controller.predict(['hi there'])
        errors = controller.correct_predictions(0, [1,1])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_WRONG_NUM_OF_PREDICTION_VALUES.format(index=result[0]._index + 1, 
                                                                                       num=config.NUM_TARGETS_BINARY_MODEL))
        controller.clear_classification()

        result, _ = controller.predict(['hi there'])
        errors = controller.correct_predictions(0, [2])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NOT_VALID_PREDICTION.format(index=result[0]._index + 1))
    
    def test_4_3(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        _, _ = controller.predict(['hi there'])
        errors = controller.correct_predictions(0, [0])
        self.assertEqual(len(errors), 0)

    def test_4_5(self):
        controller = ClassificationController()
        _, _ = controller.predict(['hi there'])
        errors = controller.correct_predictions(0, [0])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NOT_AUTHENTICATED)

    def test_5_1(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        errors = controller.authenticate('admin1', 'admin1')
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_ALREADY_AUTHENTICATED)

    def test_5_3(self):
        controller = ClassificationController()
        errors = controller.authenticate('', 'password')
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NOT_STRING_USERNAME)

        errors = controller.authenticate(' ', 'password')
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_BLANK_USERNAME)

        errors = controller.authenticate('username', '')
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NOT_STRING_PASSWORD)

        errors = controller.authenticate('username', ' ')
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_BLANK_PASSWORD)
    
    def test_5_4(self):
        controller = ClassificationController()
        valid_username, valid_password = 'a'*19, 'a'*29
        valid_username_2, valid_password_2 = 'a'*20, 'a'*30
        too_long_username, too_long_password = 'a'*21, 'a'*31

        errors = controller.authenticate(valid_username, too_long_password)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_MAX_LENGTH_PASSWORD.format(max_length=20))

        errors = controller.authenticate(valid_username_2, too_long_password)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_MAX_LENGTH_PASSWORD.format(max_length=20))

        errors = controller.authenticate(too_long_username, valid_password)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_MAX_LENGTH_USERNAME.format(max_length=20))

        errors = controller.authenticate(too_long_username, valid_password_2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_MAX_LENGTH_USERNAME.format(max_length=20))
    
    def test_5_5(self):
        controller = ClassificationController()
        username = 'º!#=^^,.'
        password = 'password'

        errors = controller.authenticate(username, password)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NOT_ALPHANUM_USERNAME)

    def test_5_6(self):
        controller = ClassificationController()
        username = 'username'
        password = 'password'

        errors = controller.authenticate(username, password)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_AUTHENTICATION)

    def test_5_7(self):
        controller = ClassificationController()
        username = 'admin1'
        password = 'admin1'

        errors = controller.authenticate(username, password)
        self.assertEqual(len(errors), 0)

    def test_6_1(self):
        controller = ClassificationController()
        path = config.PROJECT_ROOT + '/testfiles/' + 'data.csv'
        errors = controller.train_models(config.OUTPUT_BINARY_MODEL, path)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_NOT_AUTHENTICATED)

    def test_6_3(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        path = config.PROJECT_ROOT + '/testfiles/' + 'wrong_extension.txt'
        errors = controller.train_models(config.OUTPUT_BINARY_MODEL, path)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_FILE_WRONG_EXTENSION)

    def test_6_4(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        path = config.PROJECT_ROOT + '/testfiles/' + 'data_with_bad_predictions.csv'
        errors = controller.train_models(config.OUTPUT_BINARY_MODEL, path)
        self.assertEqual(len(errors), 4)
        self.assertEqual(errors[0], config.ERROR_NOT_STRING_MESSAGE.format(index=4))
        self.assertEqual(errors[1], config.ERROR_NOT_VALID_PREDICTION.format(index=5))
        self.assertEqual(errors[2], config.ERROR_NOT_VALID_PREDICTION.format(index=6))
        self.assertEqual(errors[3], config.ERROR_MAX_LENGTH_MESSAGE.format(index=7, max_length=500))

    def test_6_5(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        path = config.PROJECT_ROOT + '/testfiles/' + 'wrong_num_of_data.csv'
        errors = controller.train_models(config.OUTPUT_BINARY_MODEL, path)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], config.ERROR_FILE_WRONG_NUM_OF_COLS)
    
    def test_6_6(self):
        controller = ClassificationController()
        controller.authenticate('admin1', 'admin1')
        path = config.PROJECT_ROOT + '/testfiles/' + 'data.csv'
        errors = controller.train_models(config.OUTPUT_BINARY_MODEL, path)
        self.assertEqual(len(errors), 0)
    

if __name__ == "__main__":
    unittest.main()
