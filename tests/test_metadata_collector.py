import unittest

from text_to_sql.database.metadata_collector import MetadataCollector
from text_to_sql.database.db_engine import MySQLEngine


class TestMetadataCollector(unittest.TestCase):
    """
    To test the MetadataCollector class (get tables & columns metadata)
    """

    @classmethod
    def setUpClass(cls):
        test_engine = MySQLEngine()
        cls.metadata_collector = MetadataCollector(test_engine)

    def test_get_db_metadata(self):
        tables = self.metadata_collector.get_db_metadata()
        self.assertTrue(len(tables) > 0)
        self.assertTrue("jk_user" in tables)
        print(tables)

    def test_get_table_metadata(self):
        test_table = "jk_user"
        table_meta = self.metadata_collector.get_table_metadata(test_table)
        self.assertTrue(len(table_meta) > 0)
        print(table_meta)


if __name__ == "__main__":
    unittest.main()
