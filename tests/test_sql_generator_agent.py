import unittest
from typing import Any

from langchain_community.utilities.sql_database import SQLDatabase

from text_to_sql.llm.embedding_proxy import EmbeddingProxy
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLGeneratorAgent(unittest.TestCase):
    def setUp(self):
        self.sql_generator_agent = SQLGeneratorAgent(llm_proxy=LLMProxy(), embedding_proxy=EmbeddingProxy())

    def test_langchain_agent(self):
        pass
        from langchain.chains import create_sql_query_chain
        from sqlalchemy import create_engine

        # create a langchain database engine (SQLAlchemy)
        lang_db = SQLDatabase(
            engine=create_engine(self.sql_generator_agent.db_metadata_manager.db_engine.get_connection_url())
        )

        sql_chain = create_sql_query_chain(llm=self.sql_generator_agent.llm_proxy.llm, db=lang_db)

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

    def test_sql_agent_with_tools_chinese(self):
        question = "找到发帖数量最多的用户"
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
