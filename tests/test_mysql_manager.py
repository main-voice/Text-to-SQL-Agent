import unittest

from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.database.db_config import DBConfig


class TestMySQLEngine(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
