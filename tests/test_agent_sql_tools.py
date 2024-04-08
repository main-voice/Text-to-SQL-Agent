import unittest

from text_to_sql.database.db_config import DBConfig
from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.embedding_proxy import EmbeddingProxy
from text_to_sql.sql_generator.sql_agent_tools import (
    CurrentTimeTool,
    RelevantColumnsInfoTool,
    RelevantTablesTool,
    TablesSchemaTool,
)


class TestAgentSQLTools(unittest.TestCase):
    """
    Test the SQL agent tools. To see if the tools can achieve the expected results.
    Test tools separately. Not integrated with the whole system.
    """

    @classmethod
    def setUpClass(cls):
        cls.db_metadata_manager = DBMetadataManager(MySQLEngine(DBConfig()))
        cls.embedding = EmbeddingProxy(embedding_source="huggingface").get_embedding()

    def test_get_relevant_tables_tool(self):
        tool = RelevantTablesTool(db_manager=self.db_metadata_manager, embedding=self.embedding)
        qa_pairs = [
            ("Find the user who posted the most number of posts.", ["jk_user", "jk_post"]),
            ("Find the user whose name contains 'test'.", ["jk_user"]),
        ]
        for qa_pair in qa_pairs:
            # Only select expected number of tables
            tool.top_k = len(qa_pair[1])
            result = tool._run(qa_pair[0], remove_prefix=False)
            print(result)
            self.assertEqual(result, qa_pair[1])

    def test_get_columns_info_tool(self):
        tool = RelevantColumnsInfoTool(db_manager=self.db_metadata_manager)
        test_columns = "jk_user -> id, username, summary; jk_post -> author_id, is_deleted"
        result = tool._run(test_columns)
        assert result is not None, f"tables info of {test_columns} is None"
        print(result)

    def test_get_tables_schema_tool(self):
        tool = TablesSchemaTool(db_manager=self.db_metadata_manager)
        test_tables = "jk_user"
        result = tool._run(test_tables)
        print(result)

    def test_get_current_time_tool(self):
        """
        test CurrentTimeTool
        """
        time_tool = CurrentTimeTool(db_manager=self.db_metadata_manager)
        try:
            result = time_tool._run()
            print(result)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    unittest.main()
