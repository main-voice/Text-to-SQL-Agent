import os
import unittest

from dotenv import load_dotenv

from text_to_sql.database.db_config import MySQLConfig, PostgreSQLConfig


class TestDBConfig(unittest.TestCase):
    """
    Test the DBConfig class, mainly to check if the class read the environment variables correctly
    """

    def setUp(self) -> None:
        self.test_env_path = ".test.env"
        self.test_env_abs_path = os.path.join(os.path.dirname(__file__), self.test_env_path)

        load_dotenv(dotenv_path=self.test_env_abs_path)

    def test_mapping_config(self):
        """
        Test if pydantic can read the environment variables automatically and map them to the class
        """

        mysql_config = MySQLConfig(_env_file=self.test_env_abs_path)
        assert mysql_config.db_host == os.environ.get("MY_DB_HOST")
        assert mysql_config.db_user == os.getenv("MY_DB_USER")
        assert mysql_config.db_password == os.getenv("MY_DB_PASSWORD")
        assert mysql_config.db_port == int(os.getenv("MY_DB_PORT"))
        assert mysql_config.db_name == os.getenv("MY_DB_NAME")
        assert mysql_config.db_type == "mysql"

        print(mysql_config.dict())

        pg_config = PostgreSQLConfig(_env_file=self.test_env_abs_path)
        assert pg_config.db_host == os.environ.get("PG_DB_HOST")
        assert pg_config.db_user == os.getenv("PG_DB_USER")
        assert pg_config.db_password == os.getenv("PG_DB_PASSWORD")
        assert pg_config.db_port == 5432
        assert pg_config.db_name == os.getenv("PG_DB_NAME")
        assert pg_config.db_type == "postgresql"

        print(pg_config.dict())
