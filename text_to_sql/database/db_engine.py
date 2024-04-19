"""
Connect to database
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import mysql.connector
import psycopg2

from text_to_sql.utils.logger import get_logger

from .db_config import DBConfig, MySQLConfig, PostgreSQLConfig

logger = get_logger(__name__)


class DBEngine(ABC):
    """
    An abstract class for database engine
    """

    def __init__(self, db_config: DBConfig):
        """Initialize the database engine with the provided configuration

        Args:
            db_config (DBConfig): The configuration for the database engine connection
        """
        self.db_config = db_config
        self.connection = None

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the connection is already connected

        Returns:
            bool: True if the connection is already connected, False otherwise
        """
        pass

    @abstractmethod
    def connect(self, db_name: str = None):
        """Connect to the database using the provided configuration in __init__ method

        Args:
            db_name (str, optional): the database name to be connected. Defaults to var in env variable.
        """
        pass

    @abstractmethod
    def available_databases(self) -> List[str]:
        """Return a list of available databases in the connected database"""
        pass

    @abstractmethod
    def get_connection_url(self) -> str:
        """Return the template of connection URL SQLAlchemy needed for the database engine"""
        return "{dialect}+{driver}://{username}:{password}@{host}:{port}/{database}"

    @abstractmethod
    def disconnect(self):
        """Disconnect from the database to avoid resource leaks"""
        pass

    @abstractmethod
    def execute(
        self, statement: str, db_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query for the database, return the result as a list of dictionaries"""
        pass


class MySQLEngine(DBEngine):
    """
    MySQL database engine
    """

    def __init__(self, db_config: MySQLConfig):
        super().__init__(db_config)

    def is_connected(self) -> bool:
        return self.connection is not None and self.connection.is_connected()

    def available_databases(self) -> List[str]:
        self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute("SHOW DATABASES;")
            return [row[0] for row in cursor.fetchall()]

    def get_connection_url(self) -> str:
        mysql_template = super().get_connection_url()
        return mysql_template.format(
            dialect=self.db_config.db_type,
            driver=self.db_config.db_driver,
            username=self.db_config.db_user,
            password=self.db_config.db_password,
            host=self.db_config.db_host,
            port=self.db_config.db_port,
            database=self.db_config.db_name,
        )

    def connect(self, db_name: str = None):
        """Connect to the MySQL database using the provided configuration

        Raises:
            e: mysql.connector.InterfaceError, mysql.connector.DatabaseError, mysql.connector.Error

        Returns:
            connection: The connection object to the MySQL database
        """

        if self.is_connected():
            return self.connection

        try:
            db_name = db_name or self.db_config.db_name
            self.connection = mysql.connector.connect(
                host=self.db_config.db_host,
                user=self.db_config.db_user,
                password=self.db_config.db_password,
                port=self.db_config.db_port,
                database=db_name if db_name else "",
            )
            logger.info(f"Connected to MySQL database {self.db_config.db_name} successfully")
            return self.connection
        except (mysql.connector.InterfaceError, mysql.connector.DatabaseError, mysql.connector.Error) as e:
            logger.error(f"Failed to connect to MySQL database {self.db_config.db_name}: {e}")
            raise e

    def disconnect(self):
        if self.is_connected():
            self.connection.close()
            logger.info(f"Disconnected from MySQL database {self.db_config.db_name}")

    def execute(
        self, statement: str, db_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query to the database
        """
        if db_name:
            self.connect(db_name)
        else:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                if statement.lower().startswith("select"):
                    cursor.execute(statement, params)
                    logger.info(f"Executing SELECT query '{statement}'")
                    columns = [col[0] for col in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]

                supported_query = ["SELECT"]
                raise ValueError(f"Only {supported_query} queries are supported for now! But got {statement}")
        except Exception as e:
            logger.error(f"Failed to execute sql statement {statement}, error is {e}")
            raise e
        finally:
            if cursor:
                cursor.close()


class PostgreSQLEngine(DBEngine):
    """
    Postgresql database engine
    """

    def __init__(self, db_config: PostgreSQLConfig):
        super().__init__(db_config)

    def is_connected(self) -> bool:
        return self.connection is not None and self.connection.closed == 0

    def get_connection_url(self) -> str:
        postgres_template = super().get_connection_url()
        return postgres_template.format(
            dialect=self.db_config.db_type,
            driver=self.db_config.db_driver,
            username=self.db_config.db_user,
            password=self.db_config.db_password,
            host=self.db_config.db_host,
            port=self.db_config.db_port,
            database=self.db_config.db_name,
        )

    def available_databases(self) -> List[str]:
        self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT datname FROM pg_database;")
            return [row[0] for row in cursor.fetchall()]

    def connect(self, db_name: str = None):
        if self.is_connected():
            return self.connection

        try:
            db_name = db_name or self.db_config.db_name
            self.connection = psycopg2.connect(
                host=self.db_config.db_host,
                user=self.db_config.db_user,
                password=self.db_config.db_password,
                port=self.db_config.db_port,
                database=db_name,
            )
            logger.info(f"Connected to PostgreSQL database {self.db_config.db_name} successfully")
            return self.connection

        except (psycopg2.InterfaceError, psycopg2.OperationalError, psycopg2.DatabaseError, psycopg2.Error) as e:
            logger.error(f"Failed to connect to PostgreSQL database {self.db_config.db_name}: {e}")
            raise e

    def disconnect(self):
        if self.is_connected():
            self.connection.close()
            logger.info(f"Disconnected from PostgreSQL database {self.db_config.db_name}")

    def execute(
        self, statement: str, db_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        try:
            if db_name:
                self.connect(db_name)
            else:
                self.connect()
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL database {self.db_config.db_name}: {e}")
            raise e

        try:
            with self.connection.cursor() as cursor:
                if statement.lower().startswith("select"):
                    cursor.execute(statement, params)
                    logger.info(f"Executing SELECT query '{statement}'")
                    columns = [col[0] for col in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]

                supported_query = ["SELECT"]
                raise ValueError(f"Only {supported_query} queries are supported for now! But got {statement}")
        except Exception as e:
            logger.error(f"Failed to execute sql statement {statement}, error is {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
