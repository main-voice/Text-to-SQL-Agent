from typing import Union
from abc import ABC

import mysql.connector
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection

from text_to_sql.utils.logger import get_logger
from .db_config import DBConfig

logger = get_logger(__name__)


class DBEngine(ABC):
    """
    A class to interact with database
    """

    db_config: DBConfig
    db_connection: None
    is_connected: bool = False  # Store the status of the connection

    def __init__(self, config: DBConfig = None):
        self.db_config = config if config else DBConfig()

    def create_connection(self):
        raise NotImplementedError("create_connection method is not implemented")

    def get_connection_url(self):
        raise NotImplementedError("get_connection_url method is not implemented")

    def close_connection(self):
        raise NotImplementedError("close_connection method is not implemented")

    def execute_query(self, query):
        raise NotImplementedError("execute_query method is not implemented")


class MySQLEngine(DBEngine):
    """
    A class to interact with MySQL database
    """

    # actually type of db_connection is MySQLConnection, but we use Union to avoid warning
    db_connection: Union[PooledMySQLConnection | MySQLConnectionAbstract]

    def __init__(self, config: DBConfig = None):
        super().__init__(config)

    def create_connection(self):
        # By default, we use the info in the .env file as the name of the database
        try:
            self.db_connection = mysql.connector.connect(
                host=self.db_config.db_host,
                user=self.db_config.db_user,
                password=self.db_config.db_password,
                database=self.db_config.db_name,
            )
            logger.info(f"Connect to {self.db_config.db_name} successfully")
            self.is_connected = True
            return self.db_connection

        except mysql.connector.Error as err:
            logger.error(f"Error when connecting {self.db_config.db_name}: {err}")
            raise err

    def get_connection_url(self):
        if self.db_config.db_type == "mysql":
            default_port = 3306
            return (
                f"{self.db_config.db_type}+mysqlconnector://{self.db_config.db_user}:{self.db_config.db_password}"
                f"@{self.db_config.db_host}:{default_port}/{self.db_config.db_name}"
            )

    def close_connection(self):
        """
        Close the database connection to avoid resource leaks
        """
        # Check if the connection is already closed
        if not self.is_connected:
            return

        try:
            self.db_connection.close()
            self.is_connected = False
            logger.info(f"Close connection to {self.db_config.db_name} successfully")
        except mysql.connector.Error as err:
            logger.error(f"Error when closing connection to {self.db_config.db_name}: {err}")
            raise err

    def execute_query(self, query):
        """
        Execute a query to the database
        """
        # Check if the connection is already connected
        if not self.is_connected:
            self.create_connection()

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        query = query.strip().lower()

        cursor = self.db_connection.cursor()

        try:
            if query.lower().startswith("select"):
                logger.info(f"Executing SELECT query '{query}'")
                cursor.execute(query)
                return cursor.fetchall()
            else:
                supported_query = "SELECT"
                raise ValueError(f"Only {supported_query} queries are supported for now!")
        except mysql.connector.Error as err:
            logger.error(f"Error when executing query: {err}")
            raise err
        finally:
            # Firstly, close the cursor
            cursor.close()
            # Close database connection after executing the query
            self.close_connection()
