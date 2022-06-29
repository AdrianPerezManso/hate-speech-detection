import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_NAME = 'users.db'
QUERY_GET_USER_BY_USERNAME_AND_PASSWORD = 'SELECT password FROM users WHERE username = ?'
USERNAME_FIELD = 'username'
PASSWORD_FIELD = 'password'