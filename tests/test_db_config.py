import os
import unittest

from text_to_sql.database.db_config import MySQLConfig, PostgreSQLConfig


class TestDBConfig(unittest.TestCase):
    """
    Test the DBConfig class, mainly to check if the class read the environment variables correctly
    """

    def setUp(self) -> None:
        self.test_env_path = ".test.env.text"
        self.test_env_abs_path = os.path.join(os.path.dirname(__file__), self.test_env_path)

    def test_mapping_config(self):
        """
        Test if pydantic can read the environment variables automatically and map them to the class
        """

        mysql_config = MySQLConfig(_env_file=self.test_env_abs_path)
        assert mysql_config.db_host is not None
        assert mysql_config.db_user is not None
        assert mysql_config.db_password is not None
        assert mysql_config.db_port == 3306
        assert mysql_config.db_name is not None
        assert mysql_config.db_type == "mysql"

        print(mysql_config.dict())

        pg_config = PostgreSQLConfig(_env_file=self.test_env_abs_path)
        assert pg_config.db_host is not None
        assert pg_config.db_user is not None
        assert pg_config.db_password is not None
        assert pg_config.db_port == 5432
        assert pg_config.db_name is not None
        assert pg_config.db_type == "postgresql"

        print(pg_config.dict())
