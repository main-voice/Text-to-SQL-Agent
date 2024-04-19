import unittest

from text_to_sql.database.db_config import MySQLConfig, PostgreSQLConfig
from text_to_sql.database.db_engine import MySQLEngine, PostgreSQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager


class TestDBMetadataManager(unittest.TestCase):
    """
    To test the DBMetadataManager class (get tables & columns metadata)
    """

    @classmethod
    def setUpClass(cls):
        cls.mysql_test_engine = MySQLEngine(MySQLConfig())
        cls.mysql_manager = DBMetadataManager(cls.mysql_test_engine)

        cls.postgresql_test_engine = PostgreSQLEngine(PostgreSQLConfig(db_name="broker"))
        cls.postgresql_manager = DBMetadataManager(cls.postgresql_test_engine)

    def test_mysql_db_manager(self):
        # test if the table is in the database
        db_metadata = self.mysql_manager.get_db_metadata()
        tables_name = [table.table_name for table in db_metadata.tables]
        self.assertTrue("jk_user" in tables_name)

        # test if the column is in the table
        test_table = "jk_user"
        table_meta = self.mysql_manager.get_table_metadata(test_table)
        columns_name = [column.column_name for column in table_meta.columns]
        self.assertTrue("id" in columns_name) and self.assertTrue("username" in columns_name)

    def test_postgresql_db_manager(self):
        # test if the table is in the database
        db_metadata = self.postgresql_manager.get_db_metadata()
        tables_name = [table.table_name for table in db_metadata.tables]
        self.assertTrue("sbcustomer" in tables_name)

        # test if the column is in the table
        test_table = "sbcustomer"
        table_meta = self.postgresql_manager.get_table_metadata(test_table)
        columns_name = [column.column_name for column in table_meta.columns]
        self.assertTrue("sbcustid" in columns_name) and self.assertTrue("sbcustname" in columns_name)


if __name__ == "__main__":
    unittest.main()
