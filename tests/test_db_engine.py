import unittest

import mysql.connector
import psycopg2

from text_to_sql.database.db_config import DBConfig, MySQLConfig, PostgreSQLConfig
from text_to_sql.database.db_engine import DBEngine, MySQLEngine, PostgreSQLEngine


class TestDatabaseEngine(unittest.TestCase):
    """
    Class to test Engine to connect to the database
    """

    @classmethod
    def setUpClass(cls):
        # MySQL test data
        cls.mysql_test_db_name = "jk_space"
        cls.mysql_test_query = r"select username from jk_user"
        cls.mysql_expected_result = "baokker"

        # Postgres SQL test data
        cls.pg_test_db_name = "broker"
        cls.pg_test_query = "SELECT COUNT(sbCustId) FROM sbCustomer WHERE (LOWER(sbCustName) LIKE 'j%' OR \
        LOWER(sbCustName) LIKE '%ez') AND LOWER(sbCustState) LIKE '%a'"
        cls.pg_expected_result = 2

    def test_mysql_engine_normal(self):
        """
        test normal usage of MySQL engine
        """

        mysql_db_config = MySQLConfig(db_name=self.mysql_test_db_name)
        mysql_engine = MySQLEngine(db_config=mysql_db_config)

        mysql_engine.connect_db()
        result = mysql_engine.execute(self.mysql_test_query)
        print(result)
        mysql_engine.disconnect()

        assert self.mysql_expected_result in str(result)

    def test_postgresql_engine_normal(self):
        """
        test normal usage of Postgres SQL engine
        """
        pg_db_config = PostgreSQLConfig(db_name=self.pg_test_db_name)
        pg_engine = PostgreSQLEngine(db_config=pg_db_config)

        # pg_engine.connect()
        result = pg_engine.execute(self.pg_test_query)
        print(result)
        pg_engine.disconnect()

        assert self.pg_expected_result == result[0]["count"]

    def test_no_db_name(self):
        mysql_engine = MySQLEngine(MySQLConfig())
        mysql_engine.execute(self.mysql_test_query)

        pg_engine = PostgreSQLEngine(PostgreSQLConfig())
        pg_engine.execute(self.pg_test_query, db_name=self.pg_test_db_name)

    def test_not_select_query(self):
        mysql_engine = MySQLEngine(MySQLConfig())
        test_insert_sql = (
            "INSERT INTO jk_space.oss_auth (id, endpoint, key_id, key_secret) VALUES (12345, 'test', 'test', 'test')"
        )
        with self.assertRaises(ValueError):
            mysql_engine.execute(test_insert_sql)

        pg_engine = PostgreSQLEngine(PostgreSQLConfig(db_name=self.pg_test_db_name))
        test_insert_sql = (
            "INSERT INTO broker.sbCustomer (sbCustId, sbCustName, sbCustState) VALUES (12345, 'test', 'test')"
        )
        with self.assertRaises(ValueError):
            pg_engine.execute(test_insert_sql)

    def test_base_engine(self):
        """
        Try to use the base engine to see if it raises NotImplementedError
        """
        with self.assertRaises(TypeError):
            base_engine = DBEngine(MySQLConfig())
            base_engine.connect()

        with self.assertRaises(ValueError or TypeError):
            DBConfig()

    def test_connection_error(self):
        """
        Test if the connection error is raised
        """
        mysql_engine = MySQLEngine(MySQLConfig(db_host="localhost", db_user="non_existent_user"))
        with self.assertRaises(mysql.connector.Error):
            mysql_engine.connect_db()

        pg_engine = PostgreSQLEngine(PostgreSQLConfig(db_host="localhost", db_user="non_existent_user"))
        with self.assertRaises(psycopg2.DatabaseError or psycopg2.InterfaceError or psycopg2.Error):
            pg_engine.connect_db()

    def test_close_connection(self):
        """
        Test close the database engine connection without creating it
        """
        mysql_engine = MySQLEngine(MySQLConfig())
        mysql_engine.disconnect()

        pg_engine = PostgreSQLEngine(PostgreSQLConfig())
        pg_engine.disconnect()


if __name__ == "__main__":
    unittest.main()
