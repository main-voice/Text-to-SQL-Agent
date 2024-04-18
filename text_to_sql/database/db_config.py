"""Configuration class for the database connection"""

from pydantic import BaseSettings, Field


class DBConfig(BaseSettings):
    """
    Store the configuration info of the database
    This is base class, should not be used directly
    """

    # necessary info
    db_host: str
    db_user: str
    db_password: str

    # optional info
    db_name: str = None
    db_port: int = None
    db_type: str = None

    class Config:
        """Config for pydantic class"""

        env_path = "../config/.env"
        env_file_encoding = "utf-8"


class MySQLConfig(DBConfig):
    """
    Store the configuration info of the MySQL database
    """

    db_type: str = Field(default="mysql", const=True)
    db_port: int = Field(default=3306)

    class Config:
        """Config for pydantic class"""

        # add prefix for mysql related environment variables
        env_prefix = "MY_"


class PostgreSQLConfig(DBConfig):
    """
    Store the configuration info of the PostgreSQL database
    """

    db_type: str = Field(default="postgresql", const=True)
    db_port: int = Field(default=5432)

    class Config:
        """Config for pydantic class"""

        # add prefix for postgresql related environment variables
        env_prefix = "PG_"
