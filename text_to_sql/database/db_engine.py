"""
Connect to database
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import mysql.connector

from text_to_sql.utils.logger import get_logger

from .db_config import DBConfig

logger = get_logger(__name__)


class DBEngine(ABC):
    """
    An abstract class for database engine
    """

    def __init__(self, db_config: DBConfig):
        self.db_config = db_config
        self.connection = None
        self.is_connected = False  # Flag to check if the database is already connected

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_connection_url(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def execute(self, statement: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass


class MySQLEngine(DBEngine):
    """
    MySQL database engine
    """

    def connect(self):
        # By default, we use the info in the .env file as the name of the database
        try:
            self.connection = mysql.connector.connect(
                host=self.db_config.db_host,
                user=self.db_config.db_user,
                password=self.db_config.db_password,
                database=self.db_config.db_name,
            )
            self.is_connected = True
            logger.info(f"Connected to MySQL database {self.db_config.db_name} successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL database {self.db_config.db_name}: {e}")
            raise e

    def get_connection_url(self):
        if self.db_config.db_type == "mysql":
            default_port = 3306
            return (
                f"{self.db_config.db_type}+mysqlconnector://{self.db_config.db_user}:{self.db_config.db_password}"
                f"@{self.db_config.db_host}:{default_port}/{self.db_config.db_name}"
            )

    def disconnect(self):
        """
        Close the database connection to avoid resource leaks
        """
        # Check if the connection is already closed
        if self.connection:
            self.connection.close()
            self.is_connected = False
            logger.info(f"Disconnected from MySQL database {self.db_config.db_name}")

    def execute(self, statement: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a query to the database
        """
        # Check if the connection is already connected
        if self.is_connected is False:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(statement, params)
                if statement.lower().startswith("select"):
                    logger.info(f"Executing SELECT query '{statement}'")
                    columns = [col[0] for col in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    supported_query = ["SELECT"]
                    raise ValueError(f"Only {supported_query} queries are supported for now! But got {statement}")
                    # self.connection.commit()
                    # return []
        except Exception as e:
            logger.error(f"Failed to execute sql statement {statement}, error is {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
