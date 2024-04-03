import re
from typing import Any

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain_community.callbacks import get_openai_callback

from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm import LLMProxy, EmbeddingProxy
from text_to_sql.sql_generator.sql_agent_tools import SQLAgentToolkits
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.prompt import (
    SYSTEM_PROMPT_DEPRECATED,
    SYSTEM_CONSTRAINTS,
    DB_INTRO,
    SQL_AGENT_PREFIX,
    SQL_AGENT_SUFFIX,
    FORMAT_INSTRUCTIONS,
    SIMPLE_PLAN,
)

logger = get_logger(__name__)


class SQLGeneratorAgent:
    """
    A LLM Agent that generates SQL using user input and table metadata
    """

    def __init__(self, db_metadata_manager: DBMetadataManager, llm_proxy: LLMProxy, embedding_proxy: EmbeddingProxy):
        self.db_metadata_manager = db_metadata_manager
        self.llm_proxy = llm_proxy
        # TODO: Update the system prompt to use the SQL agent
        self.sys_prompt = SYSTEM_PROMPT_DEPRECATED
        self.embedding_proxy = embedding_proxy

    def create_sql_agent(self, verbose=True) -> AgentExecutor:
        """
        Create a SQL agent executor using our custom SQL agent tools and LLM
        """
        logger.info("Creating SQL agent executor...")

        # prepare embedding model, the embedding type is from Azure or Huggingface
        embedding = self.embedding_proxy.get_embedding()

        agent_tools = SQLAgentToolkits(db_manager=self.db_metadata_manager, embedding=embedding).get_tools()
        tools_name = [tool.name for tool in agent_tools]

        logger.info(f"The agent tools are: {tools_name}")

        # create LLM chain
        prefix = SQL_AGENT_PREFIX.format(plan=SIMPLE_PLAN)
        prompt = ZeroShotAgent.create_prompt(
            tools=agent_tools, prefix=prefix, suffix=SQL_AGENT_SUFFIX, format_instructions=FORMAT_INSTRUCTIONS
        )
        llm_chain = LLMChain(llm=self.llm_proxy.llm, prompt=prompt, verbose=verbose)

        # create sql agent executor
        sql_agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tools_name)
        sql_agent_executor = AgentExecutor.from_agent_and_tools(agent=sql_agent, tools=agent_tools)

        logger.info(f"Finished creating SQL agent executor.")
        return sql_agent_executor

    def generate_sql_with_agent(self, user_query: str, single_line_format: bool = False, verbose=True) -> Any:
        """
        Generate SQL statement using custom SQL agent executor
        """
        # create an agent executor
        sql_agent_executor = self.create_sql_agent(verbose=verbose)
        sql_agent_executor.return_intermediate_steps = True

        _input = {
            "input": user_query,
        }

        with get_openai_callback() as cb:
            try:
                response = sql_agent_executor.invoke(_input)
            except Exception as e:
                logger.error(f"Failed to generate SQL statement using SQL agent executor. Error: {e}")
                return ""
            if verbose:
                print(cb)

        if not response:
            logger.error("Failed to generate SQL statement using SQL agent executor.")
            return ""

        generated_sql = ""
        if "```sql" in response["output"]:
            generated_sql = self.extract_sql_from_llm_response(response["output"])
            generated_sql = self.remove_markdown_format(generated_sql)
        if single_line_format:
            generated_sql = self.format_sql(generated_sql)

        return generated_sql

    def generate_sql(self, user_query: str, single_line_format: bool = False) -> str:
        """
        Generate SQL statement using user input and table metadata

        :param user_query: str - The user's query, in natural language
        :param single_line_format: bool - Whether to return the SQL query as a single line or not
        :return: str - The generated SQL query
        """
        # TODO: mark this method as deprecated

        # Get tables info from metadata manager
        tables_info = self.db_metadata_manager.get_db_metadata().tables
        tables_info_json = [str(table) for table in tables_info]

        # Generate SQL statement using LLM proxy
        question = self.sys_prompt.format(
            metadata=tables_info_json, user_input=user_query, system_constraints=SYSTEM_CONSTRAINTS, db_intro=DB_INTRO
        )

        response = self.llm_proxy.get_response_from_llm(question=question).content

        if not response.startswith("```sql"):
            logger.warning("Generated SQL statement is not in the expected format, trying to extract SQL...")
            response = self.extract_sql_from_llm_response(response)

        # Extract the SQL statement from the response
        sql_statement = self.remove_markdown_format(response)

        # Perform basic validation of the SQL statement
        if (
            not sql_statement.startswith("select")
            and not sql_statement.startswith("insert")
            and not sql_statement.startswith("update")
            and not sql_statement.startswith("delete")
        ):
            logger.error("Generated SQL statement is not a SELECT, INSERT, UPDATE, or DELETE statement.")

        if single_line_format:
            return self.format_sql(sql_statement)

        return sql_statement

    # TODO: Move these formatting methods to a Output Parser
    @classmethod
    def format_sql(cls, sql) -> str:
        """
        The generated SQL statement is maybe single line or multi line. (especially for complex statement)
        Need to format it to make it easier to test
        """
        sql = sql.replace("\n", " ").replace("  ", " ").strip()
        return sql

    @classmethod
    def remove_markdown_format(cls, sql) -> str:
        """
        The LLM response contains the SQL statement in the following format:
        ```sql
        select xxxx from xxxx where xxxx
        ```
        Need to remove the markdown format to get the SQL statement
        """
        return sql.strip("```").strip("sql").strip().lower()

    @classmethod
    def extract_sql_from_llm_response(cls, sql) -> str:
        """
        The LLM response contains the SQL statement in the following format:
        ```sql
        select xxxx from xxxx where xxxx
        ```
        """
        expected_start_format = "```sql"
        expected_end_format = "```"
        expected_pattern = rf"({expected_start_format}.*?{expected_end_format})"

        # Extract the SQL statement from the response
        extracted_sql = re.findall(expected_pattern, sql, re.DOTALL)

        if not extracted_sql:
            logger.error("Failed to extract SQL statement from LLM response.")
            return ""

        if len(extracted_sql) > 1:
            logger.warning("Multiple SQL statements found in LLM response. Using the first one.")

        extracted_sql = extracted_sql[0]

        return extracted_sql
