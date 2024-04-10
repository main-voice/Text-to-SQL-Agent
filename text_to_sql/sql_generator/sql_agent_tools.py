"""
Tools for SQL Generate Agent
Tools are interfaces that an agent can use to interact with the world.

For details: ref to "https://python.langchain.com/docs/modules/agents/tools/"
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.database.models import ColumnMetadata, TableMetadata
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.translator import BaseTranslator

logger = get_logger(__name__)


class BaseSQLAgentTool(BaseModel):
    """
    Base class for SQL Agent tools.
    """

    db_manager: DBMetadataManager = Field(exclude=True)

    class Config(BaseTool.Config):
        """Config for Pydantic BaseModel"""

        arbitrary_types_allowed = True
        extra = "allow"


class RelevantTablesTool(BaseSQLAgentTool, BaseTool):
    """
    Find all possible relevant tables in the database based on user question and db metadata.
    """

    name = "DatabaseTablesWithRelevanceScores"
    description = """
        Input: User question.
        Output: A list of the most relevant tables name.
        Function: Use this tool to generate a set of tables and their relevance scores \
        to the user posed question.
        """
    embedding: Union[HuggingFaceEmbeddings, AzureOpenAIEmbeddings] = Field(exclude=True)
    top_k: int = 5  # The number of most similar tables to return, default is 5

    @property
    def tables_context(self) -> List[TableMetadata]:
        return self.db_manager.get_db_metadata().tables

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        """
        text = text.replace("\n", "").lower().strip()
        return self.embedding.embed_query(text)

    def generate_doc_embedding(self, doc: List[str]) -> List[List[float]]:
        return self.embedding.embed_documents(doc)

    @classmethod
    def cosine_similarity(cls, vec1: List[float], vec2: List[float], decimal=5) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        _cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return round(_cos, decimal)

    def _run(
        self,
        question: str,
        remove_prefix: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[str]:
        """
        Find all possible relevant tables and columns in the database based on user question and db metadata.

        Parameters:
            question: str
                The user posed question.

        Returns:
            A list of the most relevant tables name.
        """
        logger.info(f"The Agent is calling tool: {self.name}.")

        if not self.tables_context:
            raise ValueError("No table metadata found in the database. Please check the database connection.")

        question_embedding = self.generate_embedding(question)

        # Contain all tables and columns string representation
        tables_cols_str = []

        # Convert table metadata to a string
        for table in self.tables_context:
            # if remove prefix ...
            if remove_prefix:
                table_name = table.table_name.replace("jk_", "", 1)
            else:
                table_name = table.table_name

            columns_str = ""
            for column in table.columns:
                # Extract column name and comment from the column metadata to generate embedding
                if column.comment:
                    columns_str += f"{column.column_name}: {column.comment}, "
                else:
                    columns_str += f"{column.column_name}, "

            table_str = f"Table {table_name} contain columns: [{columns_str}]."
            if table.description:
                table_str += f" and table description: {table.description}"

            tables_cols_str.append([table_name, table_str])

        df = pd.DataFrame(tables_cols_str, columns=["table_name", "table_col_str"])
        df["embedding"] = self.generate_doc_embedding(df["table_col_str"].tolist())
        df["similarity"] = df["embedding"].apply(lambda x: self.cosine_similarity(question_embedding, x))

        df.sort_values(by="similarity", ascending=False, inplace=True)
        df = df.head(self.top_k)

        if df.empty:
            logger.error("No relevant tables found in the database. Please check the input question.")

        table_similarity_score = df[["table_name", "similarity"]].values.tolist()
        # format is [['table1', 0.232], ['table2', 0.123]]

        # convert to a string to make it easier to understand for LLM
        table_similarity_score_str = ""
        for item in table_similarity_score:
            table_similarity_score_str += f"Table: {item[0]}, relevance score: {item[1]}\n"

        logger.info(
            f"Found the {self.top_k} most relevant tables and their relevance scores: \
            {table_similarity_score_str} for the question: {question}."
        )

        # Now we have the table names and their relevance scores, return the most relevant tables
        # recover the table names (add jk_ prefix again)
        if remove_prefix:
            tables_name = df["table_name"].values.tolist()
            restored_tables_name = [f"jk_{table_name}" for table_name in tables_name if table_name != "oss_auth"]
            return restored_tables_name

        return df["table_name"].values.tolist()


class RelevantColumnsInfoTool(BaseSQLAgentTool, BaseTool):
    """
    Return information for given columns, including some sample rows.
    """

    name = "DatabaseRelevantColumnsInformation"
    description = """
    Input: a mapping list of tables to their columns, separated by semicolons, \
    where each table is followed by an arrow "->" and its associated columns are listed and separated by commas.

    Output: Details for the given columns in the input tables, including sample rows.

    Function: Use this tool to get more information for the potentially relevant columns. Then filter them \
    and identify those possible relevant columns based on the user posed question.

    Example input: table1 -> column1, column2; table2 -> column3, column4;
    """

    @property
    def tables_context(self) -> List[TableMetadata]:
        return self.db_manager.get_db_metadata().tables

    def _run(
        self,
        table_with_columns: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """
        Return all information for the possible relevant columns.

        Parameters:
            table_with_columns: a mapping list of tables to their columns, separated by semicolons, where each table is
            followed by an arrow "->" and its associated columns are listed and separated by commas.
            For example: table1 -> column1, column2; table2 -> column3, column4;
        """
        logger.info(f"The Agent is calling tool: {self.name}. Input table and columns name: {table_with_columns}.")

        table_column_items_list = table_with_columns.split(";")
        # remove those empty items
        table_column_items_list = [item.strip() for item in table_column_items_list if item.strip()]
        table_with_columns_dict: Dict[str, List[str]] = {}

        for item in table_column_items_list:
            if "->" not in item:
                return (
                    f"Bad input format. the item is {item}. Please refer to the Example input: "
                    "table1 -> column1, column2; table2 -> column3, column4;"
                )
            _table, _columns = item.split("->")
            table_name = _table.strip()
            columns = [_column.strip() for _column in _columns.split(",")]
            table_with_columns_dict[table_name] = columns

        if not self.tables_context:
            raise ValueError("No table metadata found in the database. Please check the database connection.")

        all_available_table_names = self.db_manager.get_available_table_names()
        if not all_available_table_names:
            raise ValueError("No table names found in the database. Please check the database connection.")

        potential_relevant_table_names: list[str] = list(table_with_columns_dict.keys())

        if potential_relevant_table_names:
            missing_tables = set(potential_relevant_table_names) - set(all_available_table_names)
            # all potential relevant tables should be a subset of all available tables
            if missing_tables:
                raise ValueError(f"Table names: {missing_tables} not found in the database. Please check the input.")

        # For below loop code, table & column variable is from Database, add type hint for better understanding
        # tables_info & column_info are string representation for the tables and columns in database
        tables_columns_info: str = ""

        for table in self.tables_context:
            if table.table_name not in potential_relevant_table_names:
                # irrelevant table, skip
                continue

            potential_relevant_column_names = table_with_columns_dict[table.table_name]

            for potential_relevant_column in potential_relevant_column_names:
                found_column: bool = False
                column_info: str = ""

                column: ColumnMetadata = self.db_manager.get_column_metadata_from_name(
                    table_name=table.table_name, column_name=potential_relevant_column
                )
                if column.column_name:
                    # The column is found in the database
                    found_column = True
                    column_info += f"column name: {column.column_name}, comment: {column.comment}.\n"
                    sample_rows = self.db_manager.get_sample_data_of_column(
                        table_name=table.table_name, column_name=column.column_name
                    )
                    column_info += f"Sample data of the column: {sample_rows}\n\n"
                    column_info = "Table: " + table.table_name + ", " + column_info

                if not found_column:
                    column_info += f"Table: {table.table_name}, column {column.column_name} not found in database.\n"

                tables_columns_info += column_info

        return tables_columns_info


class TablesSchemaTool(BaseSQLAgentTool, BaseTool):
    """
    Return tables schema for the given table names.
    """

    name = "DatabaseRelevantTablesSchema"
    description = """
    Input: A list of potentially relevant table names
    Output: Schema of the input tables.

    Function: Use this tool to get all columns information for relevant tables and \
    identify those potential columns related to user posed question.

    Example Input: table1, table2
    """

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        logger.info(f"The Agent is calling tool: {self.name}. Input table names: {table_names}.")

        table_names = table_names.split(",")
        table_names = [table_name.strip() for table_name in table_names]

        all_available_table_names = self.db_manager.get_available_table_names()
        if not all_available_table_names:
            raise ValueError("No table names found in the database. Please check the database connection.")

        if not table_names:
            missing_tables = set(all_available_table_names) - set(table_names)
            if missing_tables:
                raise ValueError(f"Table names: {missing_tables} not found in the database. Please check the input.")

        tables_schema = [self.db_manager.get_table_schema(table_name) for table_name in table_names]
        return tables_schema


class CurrentTimeTool(BaseSQLAgentTool, BaseTool):
    """
    Return the current time.
    """

    name = "CurrentTimeTool"
    description = """
    Input: an empty string
    Output: Current date and time.

    Function: Use this tool first to get the current time if there is time or date related question.
    """

    def _run(
        self,
        input_string: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        logger.info(f"The Agent is calling tool: {self.name}.")

        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        return f"The current date and time is: {current_time}"


class TranslateTool(BaseSQLAgentTool, BaseTool):
    """
    Agent tool to translate the given string to english if the given string is chinese.
    """

    name = "TranslateTool"
    description = """
    Input: an chinese string
    Output: english translation of the given string

    Function: Use this tool to translate the given string to english if the given string is chinese.
    """

    translator: BaseTranslator = Field(exclude=True)

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        logger.info(f"The Agent is calling tool: {self.name}. Input text: {text}.")

        return self.translator.translate(text)


class SQLAgentToolkits(BaseToolkit):
    """
    Return all available tools that the SQL Agent need.
    """

    db_manager: DBMetadataManager = Field(exclude=True)
    embedding: Union[HuggingFaceEmbeddings, AzureOpenAIEmbeddings] = Field(exclude=True)
    top_k: int = 5

    class Config(BaseToolkit.Config):
        """Config for Pydantic BaseModel"""

        arbitrary_types_allowed = True
        extra = "allow"

    def get_tools(self, translate_source="youdao") -> List[BaseTool]:
        # TODO: Add tools choice for the agent
        _tools = []

        # Find relevant tables tool
        relevant_tables_tool = RelevantTablesTool(
            db_manager=self.db_manager, embedding=self.embedding, top_k=self.top_k
        )
        _tools.append(relevant_tables_tool)

        # Get schema for given tables
        tables_schema_tool = TablesSchemaTool(db_manager=self.db_manager)
        _tools.append(tables_schema_tool)

        # Get information for given columns tool
        tables_info_tool = RelevantColumnsInfoTool(db_manager=self.db_manager)
        _tools.append(tables_info_tool)

        # Get current time tool
        current_time_tool = CurrentTimeTool(db_manager=self.db_manager)
        _tools.append(current_time_tool)

        # Add translation tool
        # translator: BaseTranslator = None
        # if translate_source == "llm":
        #     translator = LLMTranslator()
        # elif translate_source == "youdao":
        #     translator = YoudaoTranslator()
        #
        # _tools.append(TranslateTool(db_manager=self.db_manager, translator=translator))

        return _tools
