import unittest
import os
from src.controller.controller import ClassificationController
from src.auth.authentication import AuthenticationModule 
from src.configs import config
from src.classifiers.classifiers import BinaryModel, MLModel

class ControllerClassTest(unittest.TestCase):

    def __init__(self, controller: ClassificationController):
        self.controller = controller

    def test_correct_initialization(self):
        """Initialization test"""
        self.assertTrue(self.controller is not None)
        self.assertTrue(isinstance(self.controller.auth_module, AuthenticationModule))
        self.assertFalse(self.authenticated)
        self.assertFalse(len(self.last_predictions))

    def test_correct_change_classification_method(self):
        """Tests 1.1 and 1.2"""
        errors = self.controller.change_classification_method('wrong model')
        self.assertTrue(len(errors) == 1)
        self.assertEqual(errors[0], config.ERROR_INVALID_MODEL_OPT)
        
        errors = self.controller.change_classification_method('Binary')
        self.assertFalse(len(errors))
        self.assertTrue(isinstance(self.controller.model, BinaryModel))

        errors = self.controller.change_classification_method('Itemized')
        self.assertFalse(len(errors))
        self.assertTrue(isinstance(self.controller.model, MLModel))

    def test_correct_prediction(self):
        """Tests 2.1 to 2.8"""


    def runTest(self):
        self.test_correct_initialization()
        self.test_correct_change_classification_method()
        self.test_correct_prediction()


if __name__ == "__main__":
    unittest.main() 