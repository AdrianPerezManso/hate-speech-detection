import sqlite3
import os
from configs import config

from sqlite3 import Error

class UserRepository:
    
    def __init__(self):
        self.database_path = os.path.join(config.PROJECT_ROOT, config.DATABASE_NAME)
        self.connection = None

    def _create_connection(self):
        try:
            self.connection = sqlite3.connect(self.database_path)
        except Error as e:
            print(e)
        

    def _close_connection(self):
        try:
            if self.connection:
                self.connection.close()
        except Error as e:
            print(e)
    
    def get_password_by_username(self, usr: str):
        self._create_connection()
        try:
            cursor = self.connection.cursor()
            cursor.execute(config.QUERY_GET_USER_BY_USERNAME_AND_PASSWORD, [usr])
            result = cursor.fetchone()
            return result[0] if result else None
        except Error as e:
            print(e)
        finally:
            self._close_connection()

        
