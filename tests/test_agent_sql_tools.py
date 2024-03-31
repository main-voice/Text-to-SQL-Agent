import unittest

from text_to_sql.database.db_config import DBConfig
from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.llm.llm_proxy import get_huggingface_embedding
from text_to_sql.sql_generator.sql_agent_tools import RelevantTablesTool, InfoTablesTool
from text_to_sql.database.db_metadata_manager import DBMetadataManager


class TestAgentSQLTools(unittest.TestCase):
    """
    Test the SQL agent tools. To see if the tools can achieve the expected results.
    Test tools separately. Not integrated with the whole system.
    """

    @classmethod
    def setUpClass(cls):
        db_metadata_manager = DBMetadataManager(MySQLEngine(DBConfig()))
        cls.tables_metadata = db_metadata_manager.get_db_metadata().tables
        cls.embedding = get_huggingface_embedding()

    def test_get_relevant_tables_tool(self):
        tool = RelevantTablesTool(tables_context=self.tables_metadata, top_k=2, embedding=self.embedding)
        result = tool._run("Find the user who posted the most number of posts.")
        print(result)
        self.assertEqual(result, [["jk_user"], ["jk_post"]])

    def test_get_relevant_tables_tool_chinese(self):
        tool = RelevantTablesTool(tables_context=self.tables_metadata, top_k=2, embedding=self.embedding)
        result = tool._run("找到发帖数量最多的用户。")
        print(result)
        self.assertEqual(result, [["jk_user"], ["jk_post"]])

    def test_get_tables_info_tool(self):
        tool = InfoTablesTool(tables_context=self.tables_metadata)
        test_tables = ["jk_user", "jk_post"]
        result = tool._run(test_tables)
        print(result)
