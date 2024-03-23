from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


class SQLGeneratorAgent:
    """
    A LLM Agent that generates SQL using user input and table metadata
    """

    def __init__(self, db_metadata_manager: DBMetadataManager, llm_proxy: LLMProxy):
        self.db_metadata_manager = db_metadata_manager
        self.llm_proxy = llm_proxy
        self.sys_prompt = SYSTEM_PROMPT

    def generate_sql(self, user_query: str) -> str:
        """
        Generate SQL statement using user input and table metadata
        """
        # Get tables info from metadata manager
        tables_info = self.db_metadata_manager.get_db_metadata().tables
        tables_info_json = [str(table) for table in tables_info]

        # Generate SQL statement using LLM proxy
        question = self.sys_prompt.format(metadata=tables_info_json, user_input=user_query)
        response = self.llm_proxy.get_response_from_llm(question=question)

        # Extract the SQL statement from the response
        sql_statement = response.content.strip("```").strip("sql").strip().lower()

        # Perform basic validation of the SQL statement
        if (
            not sql_statement.startswith("select")
            and not sql_statement.startswith("insert")
            and not sql_statement.startswith("update")
            and not sql_statement.startswith("delete")
        ):
            logger.error("Generated SQL statement is not a SELECT, INSERT, UPDATE, or DELETE statement.")

        return sql_statement
