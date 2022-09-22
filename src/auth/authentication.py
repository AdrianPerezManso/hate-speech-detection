from auth.user_repository import UserRepository
from configs import logconfig
import bcrypt
import logging


class AuthenticationModule:
    """
    This class manages the authentication proccess of the application
    """
    def __init__(self):
        self.repository = UserRepository()
        logging.info(logconfig.LOG_AUTHENTICATE_INIT)

    def authenticate(self, usr: str, pwd: str):
        """
        Authentication method that communicates with the database

        :param str usr: Input username
        :param str pwd: Input password
        :return The authentication result
        :rtype bool
        """
        password = self.repository.get_password_by_username(usr)
        logging.debug(logconfig.LOG_AUTHENTICATE_EXECUTED_QUERY)
        if (not password):
            logging.info(logconfig.LOG_AUTHENTICATE_FAIL)
            return False
        return bcrypt.checkpw(pwd.encode('utf-8'), password.encode('utf-8'))


        