from text_to_sql.config.settings import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER


class DBConfig:
    """
    Store the configuration info of the database
    Use the info specified in the .env file by default
    """

    def __init__(self, db_host=None, db_name=None, db_user=None, db_password=None):
        self._db_type = "mysql"  # Only support mysql for now
        self._db_host = db_host if db_host else DB_HOST
        self._db_name = db_name if db_name else DB_NAME
        self._db_user = db_user if db_user else DB_USER
        self._db_password = db_password if db_password else DB_PASSWORD

    def __repr__(self):
        return f"DBConfig(db_type={self._db_type}, db_host={self._db_host}, db_name={self._db_name}, db_user={self._db_user})"

    @property
    # Use property to wrapper the config info
    def db_type(self):
        return self._db_type

    @property
    def db_host(self):
        return self._db_host

    @db_host.setter
    def db_host(self, value):
        self._db_host = value

    @property
    def db_name(self):
        return self._db_name

    @db_name.setter
    def db_name(self, value):
        self._db_name = value

    @property
    def db_user(self):
        return self._db_user

    @db_user.setter
    def db_user(self, value):
        self._db_user = value

    @property
    def db_password(self):
        return self._db_password

    @db_password.setter
    def db_password(self, value):
        self._db_password = value
