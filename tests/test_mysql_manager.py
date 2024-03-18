import unittest

from sql_agent.database.mysql_manager import MySQLManager


class TestMySQLManager(unittest.TestCase):
    """
    Class to test MySQLManager
    """

    @classmethod
    def setUpClass(cls):
        cls.test_selected_query = r"SELECT t.endpoint FROM jk_space.oss_auth t"
        cls.test_selected_query_expected_result = "oss-cn-shanghai.aliyuncs.com"

        cls.test_db_name = "jk_space"

    def test_no_db_name(self):
        mysql_manager = MySQLManager()
        result = mysql_manager.execute_query(self.test_selected_query)
        assert result is not None
        assert result[0][0] == self.test_selected_query_expected_result

    def test_with_db_name(self):
        mysql_manager = MySQLManager(db_name=self.test_db_name)
        result = mysql_manager.execute_query(self.test_selected_query)
        assert result is not None
        assert result[0][0] == self.test_selected_query_expected_result

    def test_not_select_query(self):
        mysql_manager = MySQLManager()
        test_insert_sql = (
            "INSERT INTO jk_space.oss_auth (endpoint, access_key_id, access_key_secret) VALUES ('test', 'test', 'test')"
        )
        with self.assertRaises(ValueError):
            mysql_manager.execute_query(test_insert_sql)


if __name__ == "__main__":
    unittest.main()
