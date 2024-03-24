import re

from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.prompt import SYSTEM_PROMPT, SYSTEM_CONSTRAINTS, DB_INTRO

logger = get_logger(__name__)


class SQLGeneratorAgent:
    """
    A LLM Agent that generates SQL using user input and table metadata
    """

    def __init__(self, db_metadata_manager: DBMetadataManager, llm_proxy: LLMProxy):
        self.db_metadata_manager = db_metadata_manager
        self.llm_proxy = llm_proxy
        self.sys_prompt = SYSTEM_PROMPT

    def generate_sql(self, user_query: str, single_line_format: bool = False) -> str:
        """
        Generate SQL statement using user input and table metadata

        :param user_query: str - The user's query, in natural language
        :param single_line_format: bool - Whether to return the SQL query as a single line or not
        :return: str - The generated SQL query
        """

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
