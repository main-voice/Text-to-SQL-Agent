import unittest
from typing import Any

from langchain_community.utilities.sql_database import SQLDatabase

from text_to_sql.llm.llm_config import LLama3LLMConfig, PerplexityLLMConfig
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLGeneratorAgent(unittest.TestCase):
    def setUp(self):
        self.sql_generator_agent = SQLGeneratorAgent()

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

    def test_sql_agent(self):
        question = "Find the user who has the most posts"
        sql = self.sql_generator_agent.generate_sql_with_agent(question, verbose=True)

        self.validate_sql(sql=sql, expected_result="baokker")

    def test_sql_agent_chinese(self):
        question = "找到发帖数量最多的用户"
        sql = self.sql_generator_agent.generate_sql_with_agent(question, verbose=True)

        self.validate_sql(sql=sql, expected_result="baokker")

    def test_sql_agent_per_llm(self):
        """
        The perplexity llm can't use custom stopping words, means we can't use the tools and ZeroShotAgent,
        so we can only use the SQLGeneratorAgent to generate SQL directly
        """
        question = "Find the user who has the most posts"
        simple_sql_agent = SQLGeneratorAgent(llm_config=PerplexityLLMConfig())
        sql = simple_sql_agent.generate_sql(user_query=question, verbose=True)
        self.validate_sql(sql=sql, expected_result="baokker")

    def test_sql_agent_llama3(self):
        """
        Test the SQLGeneratorAgent with LLama3 LLM
        """
        question = "Find the user who has the most posts"
        simple_sql_agent = SQLGeneratorAgent(llm_config=LLama3LLMConfig())
        sql = simple_sql_agent.generate_sql_with_agent(user_query=question, single_line_format=True)
        self.validate_sql(sql=sql, expected_result="baokker")

    def validate_sql(self, sql: str, expected_result: Any):
        """
        A simple way to validate the generated SQL, convert the SQL to a string directly
        and check if the expected result is in it
        """
        sql_result = self.sql_generator_agent.db_metadata_manager.db_engine.execute(sql)
        print(f"SQL result: {sql_result}")

        self.assertTrue(expected_result in str(sql_result))

    def test_preprocess(self):
        """
        Test the input preprocess function in SQL Agent
        """
        # normal usage
        question = "An normal question to test"
        result = SQLGeneratorAgent.preprocess_input(question)
        self.assertEqual(result, question)

        # input is chinese
        question_chinese = "你好，——我是。ABC！"
        result = SQLGeneratorAgent.preprocess_input(question_chinese)
        print("Input is chinese, after preprocess: " + result)
        from text_to_sql.utils import is_contain_chinese

        self.assertFalse(is_contain_chinese(result))

        # input is too long
        import random
        import string

        length = 2000
        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choices(characters, k=length))
        result = SQLGeneratorAgent.preprocess_input(random_string)
        self.assertTrue(len(result) == SQLGeneratorAgent.max_input_size)

    def test_extract_sql_from_intermediate_steps(self):
        from langchain_core.agents import AgentAction

        # extract from ValidationSQLCorrectness Tool
        intermediate_steps = [
            (
                AgentAction(
                    tool="ValidateSQLCorrectness", tool_input="```sql\nSELECT * FROM users\n```", log="test log"
                ),
                "Step 1",
            ),
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test log"), "Step 2"),
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test log"), "Step 4"),
        ]

        expected_sql = "select * from users"
        actual_sql = SQLGeneratorAgent.extract_sql_from_intermediate_steps(intermediate_steps)
        self.assertEqual(expected_sql, actual_sql)

        # failed to extract, return empty string
        intermediate_steps = [
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test log"), "Step 1"),
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test log"), "Step 2"),
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test log"), "Step 4"),
        ]
        expected_sql = ""
        actual_sql = SQLGeneratorAgent.extract_sql_from_intermediate_steps(intermediate_steps)
        self.assertEqual(expected_sql, actual_sql)

        intermediate_steps = [
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test"), "Step 1"),
            (AgentAction(tool="SomeOtherTool", tool_input="Some input", log="test log"), "Step 2"),
            (
                AgentAction(
                    tool="SomeOtherTool",
                    tool_input="Some input",
                    log="random prefix + ```sql\nSELECT * FROM users\n``` + random suffix",
                ),
                "Step 4",
            ),
        ]
        expected_sql = "select * from users"
        actual_sql = SQLGeneratorAgent.extract_sql_from_intermediate_steps(intermediate_steps)
        self.assertEqual(expected_sql, actual_sql)


if __name__ == "__main__":
    unittest.main()
