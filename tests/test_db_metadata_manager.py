import unittest

from text_to_sql.database.db_config import DBConfig
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.database.db_engine import MySQLEngine


class TestDBMetadataManager(unittest.TestCase):
    """
    To test the DBMetadataManager class (get tables & columns metadata)
    """

    @classmethod
    def setUpClass(cls):
        test_engine = MySQLEngine(DBConfig())
        cls.db_metadata_manager = DBMetadataManager(test_engine)

    def test_get_db_metadata(self):
        db_metadata = self.db_metadata_manager.get_db_metadata()
        tables_name = [table.table_name for table in db_metadata.tables]
        self.assertTrue("jk_user" in tables_name)

    def test_get_table_metadata(self):
        test_table = "jk_user"
        table_meta = self.db_metadata_manager.get_table_metadata(test_table)
        columns_name = [column.column_name for column in table_meta.columns]
        self.assertTrue("id" in columns_name) and self.assertTrue("username" in columns_name)

    def test_get_column_metadata(self):
        test_table = "jk_user"
        test_column = "username"
        column_meta = self.db_metadata_manager.get_column_metadata_from_name(test_table, test_column)
        print(column_meta)
        self.assertTrue(column_meta is not None)


if __name__ == "__main__":
    unittest.main()
