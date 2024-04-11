import unittest

from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.database.models import DBConfig
from text_to_sql.llm.embedding_proxy import EmbeddingProxy
from text_to_sql.sql_generator.sql_agent_tools import (
    CurrentTimeTool,
    RelevantColumnsInfoTool,
    RelevantTablesTool,
    TablesSchemaTool,
    ValidateSQLCorrectness,
)


class TestAgentSQLTools(unittest.TestCase):
    """
    Test the SQL agent tools. To see if the tools can achieve the expected results.
    Test tools separately. Not integrated with the whole system.
    """

    @classmethod
    def setUpClass(cls):
        # set up the info for all test cases, only execute once
        cls.db_metadata_manager = DBMetadataManager(MySQLEngine(DBConfig()))
        cls.embedding = EmbeddingProxy(embedding_source="huggingface").get_embedding()

    def test_relevant_tables_tool(self):
        tool = RelevantTablesTool(db_manager=self.db_metadata_manager, embedding=self.embedding)
        qa_pairs = [
            ("Find the user who posted the most number of posts.", ["jk_user", "jk_post"]),
            ("Find the user whose name contains 'test'.", ["jk_user"]),
        ]
        for qa_pair in qa_pairs:
            # Only select expected number of tables
            tool.top_k = len(qa_pair[1])
            result = tool._run(qa_pair[0])
            print(result)
            self.assertEqual(set(result), set(qa_pair[1]))

    def test_relevant_columns_info_tool(self):
        # normal usage
        tool = RelevantColumnsInfoTool(db_manager=self.db_metadata_manager)
        test_correct_columns = "jk_user -> id, username, summary; jk_post -> author_id, is_deleted"
        result = tool._run(test_correct_columns)
        assert result is not None, f"tables info of {test_correct_columns} is None"
        print(result)

        # the input table is fake
        test_fake_table = "fake_user -> id, username, summary;"
        with self.assertRaises(ValueError):
            tool._run(test_fake_table)

        # the input column is fake
        fake_column = "fake_column"
        assert_info = "column " + fake_column + " not found in database"
        test_fake_column = f"jk_user -> id, username, {fake_column};"
        result = tool._run(test_fake_column)
        print(result)
        self.assertTrue(assert_info in result)

    def test_table_schema_tool(self):
        tool = TablesSchemaTool(db_manager=self.db_metadata_manager)
        test_tables = "jk_user"
        result = tool._run(test_tables)
        assert result is not None, f"tables info of {test_tables} is None"
        print(result)

        fake_table = "fake_user,fake_table2"
        with self.assertRaises(ValueError):
            tool._run(fake_table)

        random_table = "DGUDHSIHDHIAIDIDKAIDIJAJIDIJIDSIDHHAMDMBVBICIASI"
        with self.assertRaises(ValueError):
            tool._run(random_table)

        wrong_type = ["ABC"]
        result = tool._run(wrong_type)
        self.assertTrue("bad input" in result.lower())

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

    def test_validate_sql_correctness_tool(self):
        """
        test ValidateSQLCorrectnessTool
        """
        # normal usage
        validate_tool = ValidateSQLCorrectness(db_manager=self.db_metadata_manager)
        test_sql = "SELECT username FROM jk_user where username like '%bao%'"
        result = validate_tool._run(test_sql)
        self.assertTrue("baokker" in result.lower())

        test_sql_markdown = "```sql" + test_sql + "```"
        result = validate_tool._run(test_sql_markdown)
        self.assertTrue("baokker" in result.lower())

        # non select sql
        test_sql = "insert into jk_user (username) values (test)"
        result = validate_tool._run(test_sql)
        # print(result)
        self.assertTrue("error" in result.lower())


if __name__ == "__main__":
    unittest.main()
