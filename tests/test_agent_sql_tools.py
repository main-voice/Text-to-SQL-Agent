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
        tool = RelevantTablesTool(tables_context=self.tables_metadata, embedding=self.embedding)
        qa_pairs = [
            ("Find the user who posted the most number of posts.", ["jk_user", "jk_post"]),
            ("Find the user whose name contains 'test'.", ["jk_user"]),
        ]
        for qa_pair in qa_pairs:
            # Only select expected number of tables
            tool.top_k = len(qa_pair[1])
            result = tool._run(qa_pair[0])
            print(result)
            self.assertEqual(result, qa_pair[1])

    def test_get_relevant_tables_tool_chinese(self):
        # TODO: NOT Passed, Add Chinese embedding support
        tool = RelevantTablesTool(tables_context=self.tables_metadata, top_k=2, embedding=self.embedding)
        result = tool._run("找到发帖数量最多的用户。")
        print(result)
        self.assertEqual(result, ["jk_user", "jk_post"])

    def test_get_tables_info_tool(self):
        tool = InfoTablesTool(tables_context=self.tables_metadata)
        test_tables = ["jk_user", "jk_post"]
        result = tool._run(test_tables)
        assert result is not None, f"tables info of {test_tables} is None"
        print(result)
