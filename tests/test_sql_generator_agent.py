import unittest

from text_to_sql.config.settings import settings
from text_to_sql.database.db_config import MySQLConfig, PostgreSQLConfig
from text_to_sql.llm.llm_config import LLama3LLMConfig, PerplexityLLMConfig, ZhiPuLLMConfig
from text_to_sql.sql_generator.sql_generate_agent import (
    BaseSQLGeneratorAgent,
    LangchainSQLGeneratorAgent,
    SimpleSQLGeneratorAgent,
    SQLGeneratorAgent,
)
from text_to_sql.utils import is_contain_chinese
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class TestSQLGeneratorAgent(unittest.TestCase):

    def test_langchain_agent(self):
        question = "Find the user who has the most posts"
        agent = LangchainSQLGeneratorAgent()
        sql = agent.generate_sql(question, instructions=None, verbose=True)
        print(f"Langchain generated SQL: {sql}")
        self.validate_sql(agent=agent, sql=sql, expected_result="baokker")

        hard_question = "How many publications were presented at each conference, ordered by the number of publications\
         in descending order? Give the names of the conferences and their corresponding number of publications."
        agent = LangchainSQLGeneratorAgent(db_config=PostgreSQLConfig())
        sql = agent.generate_sql(hard_question, instructions=None, verbose=True)

    def test_sql_agent(self):
        """Test the sql agent with Tools and ZeroShotAgent"""
        question = "Find the user who has the most posts"
        question_cn = "找到发帖数量最多的用户"

        expected_result = "baokker"

        agent = SQLGeneratorAgent(top_k=3, db_config=MySQLConfig())

        sql = agent.generate_sql(question, verbose=True)
        self.validate_sql(agent=agent, sql=sql.generated_sql, expected_result=expected_result)

        sql = agent.generate_sql(question_cn, verbose=True)
        self.validate_sql(agent=agent, sql=sql.generated_sql, expected_result=expected_result)

    def test_simple_agent_per_llm(self):
        """
        The perplexity llm can't use custom stopping words, means we can't use the tools and ZeroShotAgent,
        so we use the SimpleSQLGeneratorAgent to generate SQL directly
        """
        question = "Find the user who has the most posts"
        simple_sql_agent = SimpleSQLGeneratorAgent(llm_config=PerplexityLLMConfig())
        sql = simple_sql_agent.generate_sql(user_query=question, verbose=True)
        print(f"PerplexityLLM generated SQL: {sql}")
        self.validate_sql(agent=simple_sql_agent, sql=sql.generated_sql, expected_result="baokker")

    def test_simple_agent_llama3(self):
        """
        Test the SimpleSQLGeneratorAgent with LLama3 LLM
        """
        question = "Find the user who has the most posts"
        llama_sql_agent = SimpleSQLGeneratorAgent(llm_config=LLama3LLMConfig())
        sql = llama_sql_agent.generate_sql(user_query=question, single_line_format=True)

        print(f"LLama3 generated SQL: {sql}")
        self.validate_sql(agent=llama_sql_agent, sql=sql.generated_sql, expected_result="baokker")

    def test_simple_agent_zhipu(self):
        """Test the SimpleSQLGeneratorAgent with Zhipu LLM"""
        question = "Find the user who has the most posts"
        zhipu_sql_agent = SimpleSQLGeneratorAgent(llm_config=ZhiPuLLMConfig())
        sql = zhipu_sql_agent.generate_sql(user_query=question, single_line_format=True)

        print(f"Zhipu generated SQL: {sql}")
        self.validate_sql(agent=zhipu_sql_agent, sql=sql.generated_sql, expected_result="baokker")

    def test_sql_agent_llama3(self):
        """
        Test the SQLGeneratorAgent with LLama3 LLM
        """
        # ignore the test case for cost reason
        pass
        question = "Find the user who has the most posts"
        llama_sql_agent = SQLGeneratorAgent(db_config=MySQLConfig(), llm_config=LLama3LLMConfig())
        sql = llama_sql_agent.generate_sql(user_query=question, single_line_format=True)

        print(f"LLama3 generated SQL: {sql}")
        self.validate_sql(agent=llama_sql_agent, sql=sql.generated_sql, expected_result="baokker")

    def validate_sql(
        self, agent: BaseSQLGeneratorAgent, sql: str, expected_result: str = None, is_strict: bool = False
    ):
        """A simple way to validate the generated SQL, convert the SQL to a string directly
        and check if the expected result is in it

        Args:
            agent (BaseSQLGeneratorAgent): The SQL agent
            sql (str): the SQL to validate
            expected_result (str): The expected result
            is_strict (bool, optional): If compare the results strictly. Defaults to False.
        """
        if not is_strict or expected_result is None:
            self.assertTrue(sql is not None)
            return

        sql_result = agent.db_metadata_manager.db_engine.execute(sql)
        print(f"SQL result: {sql_result}")

        self.assertTrue(expected_result in str(sql_result))

    def test_preprocess(self):
        """
        Test the input preprocess function in SQL Agent
        """
        # normal usage
        question = "An normal question to test"
        result = BaseSQLGeneratorAgent.preprocess_input(question)
        self.assertEqual(result, question)

        # input is chinese
        question_chinese = "你好，——我是。ABC！"
        result = BaseSQLGeneratorAgent.preprocess_input(question_chinese)
        print("Input is chinese, after preprocess: " + result)

        self.assertFalse(is_contain_chinese(result))

        # input is too long
        import random
        import string

        length = 2000
        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choices(characters, k=length))
        result = BaseSQLGeneratorAgent.preprocess_input(random_string)
        self.assertTrue(len(result) == settings.MAX_INPUT_LENGTH)

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

        expected_sql = "SELECT * FROM users"
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
        expected_sql = "SELECT * FROM users"
        actual_sql = SQLGeneratorAgent.extract_sql_from_intermediate_steps(intermediate_steps)
        self.assertEqual(expected_sql, actual_sql)

    def test_handle_special_characters(self):
        sql = r"SELECT c.name AS \"Conference Name\", COUNT(pub.pid) AS \"Publication Count\" FROM conference c JOIN publication pub ON c.cid = pub.cid WHERE pub.year >= EXTRACT(YEAR FROM CURRENT_DATE) - 15 GROUP BY c.name ORDER BY COUNT(pub.pid) DESC LIMIT 1;"  # noqa
        sql = BaseSQLGeneratorAgent.format_sql(sql)
        assert "\\" not in sql
        agent = SQLGeneratorAgent(db_config=PostgreSQLConfig())
        result = agent.db_metadata_manager.db_engine.execute(sql)
        print(result)


if __name__ == "__main__":
    unittest.main()
