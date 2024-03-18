import mysql.connector
from mysql.connector import Error, MySQLConnection

from sql_agent.config.settings import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER
from sql_agent.utils.logger import get_logger

logger = get_logger(__name__)


class MySQLManager:
    """
    A class to interact with MySQL database
    """
    db_connection: MySQLConnection
    db_name: str
    is_connected: bool = False  # Store the status of the connection

    def __init__(self, db_name=None):
        self.create_connection(db_name)

    def create_connection(self, db_name):
        # By default, we use the info in the .env file as the name of the database
        self.db_name = db_name if db_name else DB_NAME

        assert (
            self.db_name is not None
        ), "database name is None, please set it in .env file or pass the name when init connection"

        try:
            self.db_connection = mysql.connector.connect(
                host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=self.db_name
            )
            logger.info(f"Connect to {self.db_name} successfully")
            self.is_connected = True
            return self.db_connection

        except mysql.connector.Error as err:
            logger.error(f"Error when connecting {self.db_name}: {err}")
            return None

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
            logger.info(f"Close connection to {self.db_name} successfully")
        except mysql.connector.Error as err:
            logger.error(f"Error when closing connection to {self.db_name}: {err}")

    def execute_query(self, query):
        """
        Execute a query to the database
        """
        # Check if the connection is already connected
        if not self.is_connected:
            self.create_connection(self.db_name)

        cursor = self.db_connection.cursor()

        try:
            if query.lower().startswith("select"):
                logger.info(f"Executing SELECT query '{query}'")
                cursor.execute(query)
                return cursor.fetchall()
            else:
                raise ValueError("Only support SELECT query for now!")
        except mysql.connector.Error as err:
            logger.error(f"Error when executing query: {err}")
            return None
        finally:
            # Firstly, close the cursor
            cursor.close()
            # Close database connection after executing the query
            self.close_connection()
