import unittest

from text_to_sql.database.db_config import DBConfig
from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.sql_generator.sql_agent_tools import RelevantTablesSQLAgent
from text_to_sql.database.db_metadata_manager import DBMetadataManager


class TestAgentSQLTools(unittest.TestCase):
    def test_get_relevant_tables(self):
        db_metadata_manager = DBMetadataManager(MySQLEngine(DBConfig()))
        tables_metadata = db_metadata_manager.get_db_metadata().tables

        sql_agent = RelevantTablesSQLAgent(tables_context=tables_metadata, top_k=2)
        result = sql_agent._run("Find the user who posted the most number of posts.")
        self.assertEqual(result, [['jk_user'], ['jk_post']])

    def test_get_relevant_tables_chinese(self):
        db_metadata_manager = DBMetadataManager(MySQLEngine(DBConfig()))
        tables_metadata = db_metadata_manager.get_db_metadata().tables

        sql_agent = RelevantTablesSQLAgent(tables_context=tables_metadata, top_k=2)
        result = sql_agent._run("找到发帖数量最多的用户。")
        self.assertEqual(result, [['jk_user'], ['jk_post']])