import unittest

import mysql.connector

from text_to_sql.database.db_config import DBConfig
from text_to_sql.database.db_engine import DBEngine, MySQLEngine


class TestDatabaseEngine(unittest.TestCase):
    """
    Class to test MySQLEngine
    """

    @classmethod
    def setUpClass(cls):
        cls.test_selected_query = r"SELECT t.endpoint FROM jk_space.oss_auth t"
        cls.test_selected_query_expected_result = "oss-cn-shanghai.aliyuncs.com"

        cls.test_db_name = "jk_space"

    def test_no_db_name(self):
        mysql_engine = MySQLEngine(DBConfig())
        result = mysql_engine.execute(self.test_selected_query)
        assert result is not None
        assert result[0]["endpoint"] == self.test_selected_query_expected_result

    def test_with_db_name(self):
        mysql_engine = MySQLEngine(DBConfig(db_name=self.test_db_name))
        result = mysql_engine.execute(self.test_selected_query)
        assert result is not None
        assert result[0]["endpoint"] == self.test_selected_query_expected_result

    def test_not_select_query(self):
        mysql_engine = MySQLEngine(DBConfig())
        test_insert_sql = (
            "INSERT INTO jk_space.oss_auth (id, endpoint, key_id, key_secret) VALUES (12345, 'test', 'test', 'test')"
        )
        with self.assertRaises(ValueError):
            mysql_engine.execute(test_insert_sql)

    def test_base_engine(self):
        """
        Try to use the base engine to see if it raises NotImplementedError
        """
        with self.assertRaises(TypeError):
            base_engine = DBEngine(DBConfig())
            base_engine.connect()

    def test_connection_error(self):
        """
        Test if the connection error is raised
        """
        mysql_engine = MySQLEngine(DBConfig(db_host="localhost", db_user="non_existent_user"))
        with self.assertRaises(mysql.connector.Error):
            test_connection = mysql_engine.connect()
            print(test_connection)

    def test_close_connection_directly(self):
        """
        Test close the database connection without creating it
        """
        mysql_engine = MySQLEngine(DBConfig())
        mysql_engine.disconnect()


if __name__ == "__main__":
    unittest.main()
