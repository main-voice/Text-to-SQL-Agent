"""
The main SQLGeneratorAgent class is responsible for generating SQL statements using a custom SQL agent executor.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Tuple

import openai
import sqlparse
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.query import create_sql_query_chain

# pylint: disable=no-name-in-module
from langchain_community.callbacks import get_openai_callback
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from sqlalchemy.exc import SQLAlchemyError

from text_to_sql.config.settings import settings
from text_to_sql.database.db_config import DBConfig, MySQLConfig, PostgreSQLConfig
from text_to_sql.database.db_engine import MySQLEngine, PostgreSQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.embedding_proxy import EmbeddingProxy
from text_to_sql.llm.llm_config import AzureLLMConfig, BaseLLMConfig
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.llm.prompts import AgentPlanPromptBuilder
from text_to_sql.sql_generator.sql_agent_tools import SQLAgentToolkits
from text_to_sql.sql_generator.sql_generate_response import SQLGeneratorResponse
from text_to_sql.utils import is_contain_chinese
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.prompt import (
    ERROR_PARSING_MESSAGE,
    FORMAT_INSTRUCTIONS,
    SIMPLE_PROMPT,
    SQL_AGENT_PREFIX,
    SQL_AGENT_SUFFIX,
)
from text_to_sql.utils.translator import LLMTranslator, YoudaoTranslator

logger = get_logger(__name__)

SQL_START_TAG = "```sql"
SQL_END_TAG = "```"
SUPPORTED_QUERYS = ["select"]

MAX_INPUT_LENGTH = settings.MAX_INPUT_LENGTH


class BaseSQLGeneratorAgent(ABC):
    """Base class for SQLGeneratorAgent"""

    llm_config: BaseLLMConfig = None
    llm_proxy: LLMProxy = None
    db_config: DBConfig = None
    db_metadata_manager: DBMetadataManager = None

    # extract sql pattern
    extract_sql_pattern = rf"({SQL_START_TAG}.*?{SQL_END_TAG})"

    def __init__(self, llm_config: BaseLLMConfig = None, db_config: DBConfig = None):
        """Create a SQLGeneratorAgent object

        Args:
            llm_config (BaseLLMConfig, optional): llm configuration. Defaults to None.
            db_config (DBConfig, optional): database configuration. Defaults to None.
        """
        # if llm_config is None, we will use the default LLM which is Azure LLM
        self.llm_config = llm_config or AzureLLMConfig()
        self.llm_proxy = LLMProxy.create_llm_proxy(config=self.llm_config)

        # if db_config is None, we will use the default MySQL database
        self.db_config = db_config or MySQLConfig()
        self.db_metadata_manager = self.create_db_metadata_manager()

    def create_db_metadata_manager(self) -> DBMetadataManager:
        """Create a database metadata manager based on the database configuration"""
        if isinstance(self.db_config, MySQLConfig):
            return DBMetadataManager(MySQLEngine(self.db_config))
        if isinstance(self.db_config, PostgreSQLConfig):
            return DBMetadataManager(PostgreSQLEngine(self.db_config))

        raise ValueError(f"Unsupported database configuration: {self.db_config.db_type}")

    @classmethod
    def preprocess_input(cls, user_query: str) -> str:
        """
        Pre-process the user input before generating SQL statement
        1. avoid too long strings
        2. translate the user input to English
        """
        if len(user_query) > MAX_INPUT_LENGTH:
            logger.warning(f"The user input is too long, trim it down to a maximum of {MAX_INPUT_LENGTH} characters.")
            user_query = user_query[:MAX_INPUT_LENGTH]

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

        return user_query

    @classmethod
    def remove_markdown_format(cls, sql: str) -> str:
        """
        The LLM response contains the SQL statement in the following format:
        ```sql
        select xxxx from xxxx where xxxx
        ```
        Need to remove the markdown format to get the SQL statement
        """
        if not sql:
            return ""

        if sql.startswith(SQL_START_TAG):
            sql = sql.replace(SQL_START_TAG, "", 1)

        if sql.endswith(SQL_END_TAG):
            sql = sql.replace(SQL_END_TAG, "", 1)

        # remove leading and trailing spaces
        sql = sql.strip()

        # sometimes LLM may response ```sql and ````
        if "`" in sql:
            sql = sql.replace("`", "")
        return sql

    @classmethod
    def extract_sql(cls, response) -> str:
        """
        Extract the SQL statement from the LLM response

        Params:
            response: The response from LLM
        Return:
            The SQL statement wrapped in ```sql and ``` tags
        """

        # Extract the SQL statement from the response
        extracted_sqls: list = re.findall(cls.extract_sql_pattern, response, re.DOTALL)

        if not extracted_sqls:
            logger.error("Failed to extract SQL statement from LLM response.")
            return ""

        if len(extracted_sqls) > 1:
            logger.warning("Multiple SQL statements found in LLM response. Using the first one.")

        extracted_sql: str = extracted_sqls[0]

        sql = cls.remove_markdown_format(extracted_sql)

        return sql

    @classmethod
    def format_sql(cls, sql: str, single_line: bool = False) -> str:
        """
        Remove unnecessary characters and format the SQL statement
        """
        if not sql:
            return ""

        sql = sql.strip()

        sql = sql.replace("\\", "")
        sql = sqlparse.format(sql, keyword_case="upper")

        if single_line:
            sql = sql.replace("\n", " ")
            sql = sql.replace("  ", " ")

        sql = str(sql)
        return sql

    @abstractmethod
    def generate_sql(
        self, user_query: str, instructions: str = None, single_line_format: bool = False, verbose: bool = True
    ) -> SQLGeneratorResponse:
        """Generate SQL statement using currect agent for the user query and instructions

        Args:
            user_query (str): The user's query, in natural language
            instructions (str): Instructions for the SQL agent when generating SQL. Defaults to None.
            single_line_format (bool, optional): If format the SQL to single line. Defaults to False.
            verbose (bool, optional): If print some additional message such as token usage. Defaults to True.
        """
        pass


class SQLGeneratorAgent(BaseSQLGeneratorAgent):
    """
    A MRKL LLM Agent that generates SQL using ZeroShotAgent in Langchain with custom SQL agent tools
    """

    def __init__(
        self,
        llm_config: BaseLLMConfig = None,
        embedding_proxy: EmbeddingProxy = None,
        db_config: DBConfig = None,
        top_k=settings.TOP_K,
        verbose=True,
        add_current_time=True,
    ):
        super().__init__(llm_config=llm_config, db_config=db_config)

        # set embedding proxy for the tools
        self.embedding_proxy = embedding_proxy or EmbeddingProxy()

        self.top_k: int = top_k
        self.verbose: bool = verbose
        self.add_current_time: bool = add_current_time

    def create_sql_agent(self, instructions: str = "", early_stopping_method: str = "generate") -> AgentExecutor:
        """
        Create a SQL agent executor using our custom SQL agent tools and LLM

        Args:
            instructions (str): Instructions for the SQL agent when generating SQL. Defaults to "".
            early_stopping_method (str): The early stopping method for the agent executor. Defaults to "generate".

        Returns:
            AgentExecutor: The SQL agent executor
        """
        logger.info("Creating SQL agent executor...")

        # prepare embedding model, the embedding type is from Azure or Huggingface
        embedding = self.embedding_proxy.get_embedding()

        # get the SQL agent tools
        sql_agent_toolkits = SQLAgentToolkits(
            db_manager=self.db_metadata_manager, embedding=embedding, top_k=self.top_k
        )
        agent_tools = sql_agent_toolkits.get_tools(with_time=self.add_current_time)
        tools_name = [tool.name for tool in agent_tools]

        logger.info(f"The agent tools are: {tools_name}")

        # create prompt for the SQL agent

        agent_plan_prompt_builder = AgentPlanPromptBuilder(db_type=self.db_config.db_type)
        agent_plan_prompt_template = agent_plan_prompt_builder.build_plan_template(
            instructions=instructions, add_current_time=self.add_current_time
        )
        agent_plan_prompt = agent_plan_prompt_template.format(instructions=instructions)

        prompt_prefix = SQL_AGENT_PREFIX.format(db_type=self.db_config.db_type, plan=agent_plan_prompt)
        prompt = ZeroShotAgent.create_prompt(
            tools=agent_tools, prefix=prompt_prefix, suffix=SQL_AGENT_SUFFIX, format_instructions=FORMAT_INSTRUCTIONS
        )

        # create LLM chain
        llm_chain = LLMChain(llm=self.llm_proxy.llm, prompt=prompt, verbose=self.verbose)

        # create sql agent executor
        sql_agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tools_name)
        if self.llm_config.llm_source in ["azure", "meta", "deepseek"]:
            sql_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sql_agent, tools=agent_tools, early_stopping_method=early_stopping_method
            )
        else:
            # for perplexity llm, it doesn't support custom stopping words
            sql_agent_executor = AgentExecutor.from_agent_and_tools(agent=sql_agent, tools=agent_tools)

        logger.info("Finished creating SQL agent executor.")
        return sql_agent_executor

    def generate_sql(
        self,
        user_query: str,
        instructions: str = None,
        single_line_format: bool = False,
        verbose: bool = True,
    ) -> SQLGeneratorResponse:
        """Generate SQL statement using the SQL agent executor

        Args:
            user_query (str): The user's query, in natural language
            instructions (str, optional): The user's instructions for the SQL. Defaults to None.
            add_current_time (bool, optional): If add current time to the prompt. Defaults to True.
            single_line_format (bool, optional): If format the SQL to a single line. Defaults to False.
            verbose (bool, optional): If logging some info. Defaults to True.

        Returns:
            SQLGeneratorResponse: The response of the SQL generator
        """

        # create an agent executor
        sql_agent_executor = self.create_sql_agent(instructions=instructions)
        sql_agent_executor.return_intermediate_steps = True
        sql_agent_executor.handle_parsing_errors = ERROR_PARSING_MESSAGE

        _user_query = self.preprocess_input(user_query)
        _input = {
            "input": _user_query,
        }

        with get_openai_callback() as cb:
            error = None
            try:
                response = sql_agent_executor.invoke(_input)
            except (openai.APIConnectionError, openai.AuthenticationError, openai.RateLimitError) as e:
                logger.error(f"OpenAI API error: {e}")
                error = str(e)
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error: {e}")
                error = str(e)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    f"Failed to generate SQL statement using SQL agent executor. Error: {e}, "
                    f"error type: {type(e).__name__}"
                )
                error = str(e)

            if error:
                return SQLGeneratorResponse(
                    question=user_query,
                    db_name=self.db_config.db_name,
                    generated_sql="",
                    token_usage=-1,
                    llm_source=self.llm_config.llm_source,
                    llm_model=self.llm_config.model,
                    error=error,
                )

            if self.verbose or verbose:
                logger.info(f"The callback from openai is:\n{cb}\n")

        generated_sql = ""
        if SQL_START_TAG in response["output"]:
            generated_sql = self.extract_sql(response["output"])
        else:
            logger.info(f"Not found {SQL_START_TAG} in LLM output, trying to find result from intermediate steps")
            generated_sql = self.extract_sql_from_intermediate_steps(response["intermediate_steps"])
            if generated_sql:
                logger.info(f"Found SQL from intermediate steps: {generated_sql}")

        generated_sql = self.format_sql(generated_sql, single_line=single_line_format)

        return SQLGeneratorResponse(
            question=user_query,
            db_name=self.db_config.db_name,
            generated_sql=generated_sql,
            token_usage=cb.total_tokens,
            llm_source=self.llm_config.llm_source,
            llm_model=self.llm_config.model,
        )

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
                sql = cls.extract_sql(tool_input)

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
                if SQL_START_TAG in thought:
                    sql = cls.extract_sql(thought)
                    if not sql.lower().strip().startswith("select"):
                        sql = ""
        return sql


class SimpleSQLGeneratorAgent(BaseSQLGeneratorAgent):
    """A simple agent without any tools to generate SQL statements"""

    def __init__(self, llm_config: BaseLLMConfig = None, db_config: DBConfig = None):
        super().__init__(llm_config=llm_config, db_config=db_config)

    def generate_sql(
        self, user_query: str, instructions: str = None, single_line_format: bool = False, verbose: bool = True
    ) -> SQLGeneratorResponse:
        # preprocess the user input
        _user_query = self.preprocess_input(user_query)

        if instructions == "nan":
            instructions = ""

        # Get tables info from metadata manager
        tables = self.db_metadata_manager.get_db_metadata().tables
        tables_info = []

        for table in tables:
            table_name = table.table_name

            columns_str = ""
            for column in table.columns:
                if column.comment:
                    columns_str += f"{column.column_name}: {column.comment}, "
                else:
                    columns_str += f"{column.column_name}, "

            table_str = f"Table {table_name} contain columns: [{columns_str}]."
            if table.description:
                table_str += f" and table description: {table.description}"

            tables_info.append([table_name, table_str])

        # Generate SQL statement using LLM proxy
        question = SIMPLE_PROMPT.format(
            dialect=self.db_config.db_type,
            database_metadata=tables_info,
            user_input=_user_query,
            instructions=instructions,
        )

        raw_response: BaseMessage = self.llm_proxy.get_response_from_llm(question=question, verbose=verbose)
        response = raw_response.content

        if SQL_START_TAG in response:
            response = self.extract_sql(response)
        else:
            logger.warning("Generated SQL statement is not in the expected format, trying to extract SQL...")
            response = self.extract_sql(str(raw_response))

        response = self.format_sql(sql=response, single_line=single_line_format)

        token_usage = -1
        if (
            "token_usage" in raw_response.response_metadata
            and "total_tokens" in raw_response.response_metadata["token_usage"]
        ):
            token_usage = max(raw_response.response_metadata["token_usage"]["total_tokens"], token_usage)

        return SQLGeneratorResponse(
            question=user_query,
            db_name=self.db_config.db_name,
            generated_sql=response,
            token_usage=token_usage,
            llm_source=self.llm_config.llm_source,
            llm_model=self.llm_config.model,
        )


class LangchainSQLGeneratorAgent(BaseSQLGeneratorAgent):
    """A SQL generator agent that uses Langchain to generate SQL statements"""

    def __init__(self, llm_config: BaseLLMConfig = None, db_config: DBConfig = None):
        super().__init__(llm_config=llm_config, db_config=db_config)

    def generate_sql(
        self, user_query: str, instructions: str = None, single_line_format: bool = False, verbose: bool = True
    ) -> SQLGeneratorResponse:
        # preprocess the user input
        _user_query = self.preprocess_input(user_query)
        _input = {"question": _user_query, "instructions": instructions}

        # create a langchain SQL chain
        langchain_db = SQLDatabase(engine=self.db_metadata_manager.db_engine.get_sqlalchemy_engine())

        # prompt = MYSQL_PROMPT if isinstance(self.db_config, MySQLConfig) else LANGCHAIN_POSTGRES_PROMPT

        sql_chain = create_sql_query_chain(llm=self.llm_proxy.llm, db=langchain_db)

        with get_openai_callback() as cb:
            try:
                response = sql_chain.invoke(_input)
                error = ""
            except (openai.APIConnectionError, openai.AuthenticationError, openai.RateLimitError) as e:
                logger.error(f"OpenAI API error: {e}")
                error = str(e)
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error: {e}")
                error = str(e)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    f"Failed to generate SQL statement using Langchain. Error: {e}, error type: {type(e).__name__}"
                )
                error = str(e)

            if error:
                return SQLGeneratorResponse(
                    question=user_query,
                    db_name=self.db_config.db_name,
                    generated_sql="",
                    token_usage=-1,
                    llm_source=self.llm_config.llm_source,
                    llm_model=self.llm_config.model,
                    error=error,
                )

        if response.lower().startswith("select"):
            response = response.split(";")[0]
        elif SQL_START_TAG in response:
            response = self.extract_sql(response)
        elif "select" in response.lower():
            # Format: SQLQuery: \n SELECT xxx FROM xxx WHERE xxx;\n\nSQLResult:
            # Extract the SQL statement from the response
            pattern = r"SELECT.*?;"
            matches = re.findall(pattern, response, re.S | re.I)
            for match in matches:
                match = match.strip()
            if len(matches) > 1:
                logger.warning("Multiple SQL statements found in Langchain response. Using the first one.")
            response = matches[0] if matches else ""
        else:
            logger.warning("Failed to extract SQL statement from Langchain response")
            response = ""

        response = self.format_sql(sql=response, single_line=single_line_format)

        return SQLGeneratorResponse(
            question=user_query,
            db_name=self.db_config.db_name,
            generated_sql=response,
            token_usage=cb.total_tokens,
            llm_source=self.llm_config.llm_source,
            llm_model=self.llm_config.model,
        )
