"""
The main SQLGeneratorAgent class is responsible for generating SQL statements using a custom SQL agent executor.
"""

import re
from typing import Any, List, Tuple

import openai
from deprecated import deprecated
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.llm import LLMChain

# pylint: disable=no-name-in-module
from langchain_community.callbacks import get_openai_callback
from langchain_core.agents import AgentAction
from sqlalchemy.exc import SQLAlchemyError

from text_to_sql.database.db_config import DBConfig, MySQLConfig, PostgreSQLConfig
from text_to_sql.database.db_engine import MySQLEngine, PostgreSQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.embedding_proxy import EmbeddingProxy
from text_to_sql.llm.llm_config import AzureLLMConfig, BaseLLMConfig
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.sql_generator.sql_agent_tools import SQLAgentToolkits
from text_to_sql.utils import is_contain_chinese
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.prompt import (
    DB_INTRO,
    ERROR_PARSING_MESSAGE,
    FORMAT_INSTRUCTIONS,
    PLAN_WITH_VALIDATION,
    SQL_AGENT_PREFIX,
    SQL_AGENT_SUFFIX,
    SYSTEM_CONSTRAINTS,
    SYSTEM_PROMPT_DEPRECATED,
)
from text_to_sql.utils.translator import LLMTranslator, YoudaoTranslator

logger = get_logger(__name__)


class SQLGeneratorAgent:
    """
    A LLM Agent that generates SQL using user input and table metadata
    """

    max_input_size: int = 200

    def __init__(
        self,
        llm_config: BaseLLMConfig = None,
        embedding_proxy: EmbeddingProxy = None,
        db_config: DBConfig = None,
        top_k=5,
        verbose=True,
    ):
        # set database metadata manager
        if db_config is None:
            # by default, we use MySQL database
            self.db_config = MySQLConfig()
            self.db_metadata_manager = DBMetadataManager(MySQLEngine(MySQLConfig()))
        else:
            self.db_config = db_config
            if isinstance(db_config, MySQLConfig):
                self.db_metadata_manager = DBMetadataManager(MySQLEngine(db_config))
            elif isinstance(db_config, PostgreSQLConfig):
                self.db_metadata_manager = DBMetadataManager(PostgreSQLEngine(db_config))

        if llm_config is None:
            # If llm_config is None, we will use the default LLM which is Azure LLM
            self.llm_config = AzureLLMConfig()
        else:
            self.llm_config = llm_config

        # set LLM proxy
        self.llm_proxy = LLMProxy(config=self.llm_config)

        # set embedding proxy
        if embedding_proxy is None:
            self.embedding_proxy = EmbeddingProxy()
        else:
            self.embedding_proxy = embedding_proxy

        self.top_k: int = top_k
        self.verbose: bool = verbose

    def create_sql_agent(self, early_stopping_method: str = "generate") -> AgentExecutor:
        """
        Create a SQL agent executor using our custom SQL agent tools and LLM
        """
        logger.info("Creating SQL agent executor...")

        # prepare embedding model, the embedding type is from Azure or Huggingface
        embedding = self.embedding_proxy.get_embedding()

        agent_tools = SQLAgentToolkits(
            db_manager=self.db_metadata_manager, embedding=embedding, top_k=self.top_k
        ).get_tools()
        tools_name = [tool.name for tool in agent_tools]

        logger.info(f"The agent tools are: {tools_name}")

        # create LLM chain
        prefix = SQL_AGENT_PREFIX.format(db_type=self.db_config.db_type, plan=PLAN_WITH_VALIDATION)
        prompt = ZeroShotAgent.create_prompt(
            tools=agent_tools, prefix=prefix, suffix=SQL_AGENT_SUFFIX, format_instructions=FORMAT_INSTRUCTIONS
        )
        llm_chain = LLMChain(llm=self.llm_proxy.llm, prompt=prompt, verbose=self.verbose)

        # create sql agent executor
        sql_agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tools_name)
        if self.llm_config.llm_source == "azure":
            sql_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sql_agent, tools=agent_tools, early_stopping_method=early_stopping_method
            )
        else:
            # for perplexity llm, it doesn't support custom stopping words
            sql_agent_executor = AgentExecutor.from_agent_and_tools(agent=sql_agent, tools=agent_tools)

        logger.info("Finished creating SQL agent executor.")
        return sql_agent_executor

    def generate_sql_with_agent(self, user_query: str, single_line_format: bool = False) -> Any:
        """
        Generate SQL statement using custom SQL agent executor
        """
        # create an agent executor
        sql_agent_executor = self.create_sql_agent()
        sql_agent_executor.return_intermediate_steps = True
        sql_agent_executor.handle_parsing_errors = ERROR_PARSING_MESSAGE

        user_query = self.preprocess_input(user_query)
        _input = {
            "input": user_query,
        }

        with get_openai_callback() as cb:
            try:
                response = sql_agent_executor.invoke(_input)
            except openai.AuthenticationError as e:
                logger.error(f"OpenAI API authentication error: {e}")
                return ""
            except openai.RateLimitError as e:
                logger.error(f"OpenAI API rate limit error: {e}")
                return f"OpenAI API rate limit error: {e}"
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error: {e}")
                return f"SQLAlchemy error: {e}"
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    f"Failed to generate SQL statement using SQL agent executor. Error: {e}, "
                    f"error type: {type(e).__name__}"
                )
                return None
            if self.verbose:
                logger.info(f"The callback from openai is:\n{cb}\n")

        if not response:
            logger.error("Failed to generate SQL statement using SQL agent executor.")
            return ""

        generated_sql = ""
        if "```sql" in response["output"]:
            generated_sql = self.extract_sql_from_llm_response(response["output"])
            generated_sql = self.remove_markdown_format(generated_sql)
        else:
            logger.info("Not found ```sql in LLM output, trying to find result from intermediate steps")
            generated_sql = self.extract_sql_from_intermediate_steps(response["intermediate_steps"])

        if single_line_format:
            generated_sql = self.format_sql(generated_sql)

        return generated_sql

    @classmethod
    def preprocess_input(cls, user_query: str) -> str:
        """
        Pre-process the user input before generating SQL statement
        1. avoid too long strings
        2. translate the user input to English
        """

        if len(user_query) > cls.max_input_size:
            logger.warning(f"The user input is too long, trim it down to a maximum of {cls.max_input_size} characters.")
            user_query = user_query[: cls.max_input_size]

        if is_contain_chinese(user_query):
            # translate the user input to English
            try:
                logger.info("Translating user input to English using Youdao...")

                translator = YoudaoTranslator()
                user_query = translator.translate(user_query)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"ERROR when translating input {user_query} to English using Youdao translate: {e}")
                logger.info("Translating user input to English using LLM...")

                translator = LLMTranslator()
                user_query = translator.translate(user_query)

            return user_query

        # don't forget here when the input is normal english statement.
        return user_query

    @deprecated(version="0.1.0", reason="This function only use simple prompt, use generate_sql_with_agent instead")
    def generate_sql(self, user_query: str, single_line_format: bool = False, verbose=True) -> str:
        """
        Generate SQL statement using user input and table metadata

        :param user_query: str - The user's query, in natural language
        :param single_line_format: bool - Whether to return the SQL query as a single line or not
        :param verbose: bool - Whether to log the token usage or not
        :return: str - The generated SQL query
        """

        # Get tables info from metadata manager
        tables_info = self.db_metadata_manager.get_db_metadata().tables
        tables_info_json = [str(table) for table in tables_info]

        # Generate SQL statement using LLM proxy
        prompt = SYSTEM_PROMPT_DEPRECATED
        question = prompt.format(
            metadata=tables_info_json, user_input=user_query, system_constraints=SYSTEM_CONSTRAINTS, db_intro=DB_INTRO
        )

        response = self.llm_proxy.get_response_from_llm(question=question, verbose=verbose).content

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
        sql = sql.strip("```").strip("sql").strip().lower()
        # sometimes LLM may response ```sql and ````
        if "`" in sql:
            sql = sql.replace("`", "")
        return sql

    @classmethod
    def extract_sql_from_llm_response(cls, response) -> str:
        """
        The LLM response contains the SQL statement in the following format:
        ```sql
        select xxxx from xxxx where xxxx
        ```

        Params:
            response: The response from LLM
        Return:
            The SQL statement wrapper by ```sql and ``` tags
        """
        expected_start_format = "```sql"
        expected_end_format = "```"
        expected_pattern = rf"({expected_start_format}.*?{expected_end_format})"

        # Extract the SQL statement from the response
        extracted_sqls: list = re.findall(expected_pattern, response, re.DOTALL)

        if not extracted_sqls:
            logger.error("Failed to extract SQL statement from LLM response.")
            return ""

        if len(extracted_sqls) > 1:
            logger.warning("Multiple SQL statements found in LLM response. Using the first one.")

        extracted_sql: str = extracted_sqls[0]

        return extracted_sql

    @classmethod
    def extract_sql_from_intermediate_steps(cls, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        """Extract the SQL statement from the agent intermediate steps"""
        sql = ""

        # trying to find sql from the input of ValidateSQLCorrectness tool
        for step in intermediate_steps:
            action = step[0]
            if isinstance(action, AgentAction) and action.tool == "ValidateSQLCorrectness":
                # Input: an SQL wrapper in ```sql and ``` tags.
                tool_input = action.tool_input
                sql = cls.extract_sql_from_llm_response(tool_input)
                sql = cls.remove_markdown_format(sql)

        # if we don't find sql in the ValidateSQLCorrectness tool, we will try to find sql in previous step
        if not sql:
            logger.info("No valid SQL in ValidateSQLCorrectness tool input, trying to find in the previous steps.")
            for step in intermediate_steps:
                action = step[0]
                # the log format is
                # Thought: xxx
                # Action: xxx
                # Action Input: xxx
                thought = action.log.split("Action:")[0]
                if "```sql" in thought:
                    sql = cls.extract_sql_from_llm_response(thought)
                    sql = cls.remove_markdown_format(sql)
                    if not sql.lower().strip().startswith("select"):
                        sql = ""
        return sql
