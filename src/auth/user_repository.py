from configs import config, logconfig
from sqlite3 import Error
import sqlite3
import os
import logging

class UserRepository:
    """
    Repository for the users' database. Manages connection and queries
    """
    def __init__(self):
        self.database_path = config.DATABASE_DIR
        self.connection = None
        logging.debug(logconfig.LOG_USER_REPO_INIT)

    def _create_connection(self):
        """
        Create connection with database
        """
        try:
            self.connection = sqlite3.connect(self.database_path)
            logging.debug(logconfig.LOG_USER_REPO_CREATED_CON)
        except Error as e:
            logging.error(logconfig.LOG_USER_REPO_ERROR).format(error=e)
        

    def _close_connection(self):
        """
        Close connection with database
        """
        try:
            if self.connection:
                self.connection.close()
                logging.debug(logconfig.LOG_USER_REPO_CLOSED_CON)
        except Error as e:
            logging.error(logconfig.LOG_USER_REPO_ERROR).format(error=e)
    
    def get_password_by_username(self, usr: str):
        """
        This method returns any password found for the username given
        
        :param str usr: The username
        :return: The password retrieved from the database. If it didn't exist None is returned
        :rtype: str or None
        """
        self._create_connection()
        try:
            cursor = self.connection.cursor()
            cursor.execute(config.QUERY_GET_USER_BY_USERNAME_AND_PASSWORD, [usr])
            result = cursor.fetchone()
            return result[0] if result else None
        except Error as e:
            logging.error(logconfig.LOG_USER_REPO_ERROR).format(error=e)
        finally:
            self._close_connection()

        
