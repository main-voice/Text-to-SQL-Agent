import unittest

import mysql.connector

from text_to_sql.database.db_engine import MySQLEngine, DBEngine
from text_to_sql.database.db_config import DBConfig


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
        mysql_engine = MySQLEngine()
        result = mysql_engine.execute_query(self.test_selected_query)
        assert result is not None
        assert result[0][0] == self.test_selected_query_expected_result

    def test_with_db_name(self):
        mysql_engine = MySQLEngine(DBConfig(db_name=self.test_db_name))
        result = mysql_engine.execute_query(self.test_selected_query)
        assert result is not None
        assert result[0][0] == self.test_selected_query_expected_result

    def test_not_select_query(self):
        mysql_engine = MySQLEngine()
        test_insert_sql = (
            "INSERT INTO jk_space.oss_auth (endpoint, access_key_id, access_key_secret) VALUES ('test', 'test', 'test')"
        )
        with self.assertRaises(ValueError):
            mysql_engine.execute_query(test_insert_sql)

    def test_base_engine(self):
        """
        Try to use the base engine to see if it raises NotImplementedError
        """
        base_engine = DBEngine()
        with self.assertRaises(NotImplementedError):
            base_engine.create_connection()

        with self.assertRaises(NotImplementedError):
            base_engine.close_connection()

        with self.assertRaises(NotImplementedError):
            base_engine.get_connection_url()

        with self.assertRaises(NotImplementedError):
            base_engine.execute_query(self.test_selected_query)

    def test_connection_error(self):
        """
        Test if the connection error is raised
        """
        mysql_engine = MySQLEngine(DBConfig(db_host="localhost", db_user="non_existent_user"))
        with self.assertRaises(mysql.connector.Error):
            test_connection = mysql_engine.create_connection()

    def test_close_connection_directly(self):
        """
        Test close the database connection without creating it
        """
        mysql_engine = MySQLEngine()
        mysql_engine.close_connection()


if __name__ == "__main__":
    unittest.main()
