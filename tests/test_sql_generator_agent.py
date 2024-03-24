import unittest
from typing import List, Tuple

from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLGeneratorAgent(unittest.TestCase):
    def setUp(self):
        self.sql_generator_agent = SQLGeneratorAgent(
            db_metadata_manager=DBMetadataManager(MySQLEngine()), llm_proxy=LLMProxy()
        )

    def test_generate_sql_single_table_with_columns(self):
        sql = self.sql_generator_agent.generate_sql("Show me the names of all the users")
        self.assertEqual(sql, "select username from jk_user;")

        sql = self.sql_generator_agent.generate_sql("Show me the users whose name is 'test'")
        self.assertEqual(sql, "select * from jk_user where username = 'test';")

        sql = self.sql_generator_agent.generate_sql("Find all users who have been authenticated with a student ID.")
        self.assertEqual(sql, "select * from jk_user where student_id is not null;")

    def test_generate_sql_single_table_chinese(self):
        # Test agent ability to understand Chinese question
        sql = self.sql_generator_agent.generate_sql("系统中有多少男性用户？")
        self.assertEqual(sql, "select count(*) from jk_user where gender = '男';")

    def test_generate_sql_multi_tables(self):
        # Multi tables query
        question = "找到发帖数量最多的用户"
        sql = self.sql_generator_agent.generate_sql(question, single_line_format=True)
        # for complex query, there is multiple ways to express the same query, so we need to run the query to check
        print(f"Generated SQL for '找到发帖数量最多的用户': {sql}")

        sql_result = self.sql_generator_agent.db_metadata_manager.db_engine.execute_query(sql)
        print(f"SQL result: {sql_result}")

        if isinstance(sql_result, List) and isinstance(sql_result[0], Tuple):
            self.assertTrue("baokker" in sql_result[0])
        else:
            logger.warning("The return type of SQL result is not List[Tuple], please check!")
            exit(1)


if __name__ == "__main__":
    unittest.main()
