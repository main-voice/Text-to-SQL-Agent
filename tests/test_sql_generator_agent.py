import unittest
from typing import Any

from langchain_community.utilities.sql_database import SQLDatabase

from text_to_sql.llm import LLMProxy, EmbeddingProxy
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLGeneratorAgent(unittest.TestCase):
    def setUp(self):
        self.sql_generator_agent = SQLGeneratorAgent(
            llm_proxy=LLMProxy(),
            embedding_proxy=EmbeddingProxy()
        )

    def test_generate_sql_single_table_with_columns(self):
        sql = self.sql_generator_agent.generate_sql("Show me the names of all the users")
        self.assertEqual(sql, "select username from jk_user;")

        sql = self.sql_generator_agent.generate_sql("Show me the users whose name is 'test'")
        self.assertEqual(sql, "select * from jk_user where username = 'test';")

        sql = self.sql_generator_agent.generate_sql(
            "Find all users' name who have been authenticated with a student ID."
        )
        self.assertEqual(sql, "select username from jk_user where student_id is not null;")

    def test_generate_sql_single_table_chinese(self):
        # Test agent ability to understand Chinese question
        sql = self.sql_generator_agent.generate_sql("系统中有多少男性用户？")
        self.assertEqual(sql, "select count(*) from jk_user where gender = '男';")

    def test_generate_sql_multi_tables(self):
        # Multi tables query
        question = "找到发帖数量最多的用户"
        sql = self.sql_generator_agent.generate_sql(question, single_line_format=True)
        # for complex query, there is multiple ways to express the same query, so we need to run the query to check
        print(f"Generated SQL for {question}: {sql}")

    def test_langchain_agent(self):
        from langchain.chains import create_sql_query_chain
        from sqlalchemy import create_engine

        # create a langchain database engine (SQLAlchemy)
        lang_db = SQLDatabase(
            engine=create_engine(self.sql_generator_agent.db_metadata_manager.db_engine.get_connection_url())
        )

        sql_chain = create_sql_query_chain(
            llm=self.sql_generator_agent.llm_proxy.llm,
            db=lang_db
        )

        # msg = {"question": "show me the names of all users"}
        # msg = {"question": "找到发帖数量最多的用户"}
        msg = {"question": "Find the user who has the most posts"}
        _sql = sql_chain.invoke(msg)
        print(f"SQL: {_sql}")
        self.validate_sql(sql=_sql, expected_result="baokker")

    def test_sql_agent_with_tools(self):
        question = "Find the user who has the most posts"
        sql = self.sql_generator_agent.generate_sql_with_agent(question, verbose=True)

        self.validate_sql(sql=sql, expected_result="baokker")

    def validate_sql(self, sql: str, expected_result: Any):
        """
        A simple way to validate the generated SQL, convert the SQL to a string directly
        and check if the expected result is in it
        """
        sql_result = self.sql_generator_agent.db_metadata_manager.db_engine.execute(sql)
        print(f"SQL result: {sql_result}")

        self.assertTrue(expected_result in str(sql_result))


if __name__ == "__main__":
    unittest.main()
