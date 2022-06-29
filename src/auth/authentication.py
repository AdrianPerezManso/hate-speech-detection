from auth.user_repository import UserRepository
import bcrypt

class AuthenticationModule:
    def __init__(self):
        self.repository = UserRepository()

    def authenticate(self, usr: str, pwd: str):
        password = self.repository.get_password_by_username(usr)
        if (not password): return False
        return bcrypt.checkpw(pwd.encode('utf-8'), password.encode('utf-8'))


        