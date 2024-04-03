"""
Tools for SQL Generate Agent
Tools are interfaces that an agent can use to interact with the world.

For details: ref to "https://python.langchain.com/docs/modules/agents/tools/"
"""

# TODO: Add a Chinese to English translation tool


from typing import List, Any, Optional, Union

import numpy as np
import pandas as pd
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.database.models import TableMetadata
from text_to_sql.utils.logger import get_logger

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
        Function: Use this tool to generate a set of tables and their relevance scores 
                  to the user posed question.
        Output: A dictionary with table names as keys and relevance scores as values.
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
            A Tuple (A, B)
            A is a string representation of the most relevant tables and their relevance scores. (for better understanding)
            B is a list of the most relevant tables name. (for better testing)
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


class TablesInfoTool(BaseSQLAgentTool, BaseTool):
    """
    Return information for given table names.
    """

    name = "DatabaseTablesInformation"
    description = """
        Input: A list of table names.
        Function: Use this tool to get all columns information for the relevant tables.
                  And identify those possible relevant columns based on the user posed question.
        Output: Metadata for the input tables.
        
        Input Example: ["table1", "table2"]
        """

    @property
    def tables_context(self) -> List[TableMetadata]:
        return self.db_manager.get_db_metadata().tables

    def _run(
        self,
        table_names: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """
        Return all available tables in the database.
        """
        if not table_names:
            raise ValueError("No table names found in the input. Please check the previous tool output.")

        if not self.tables_context:
            raise ValueError("No table metadata found in the database. Please check the database connection.")

        logger.info(f"The Agent is calling tool: {self.name}. Input table names: {table_names}.")

        # Contain all tables and columns string representation
        tables_info = ""

        for _table_meta in self.tables_context:
            if _table_meta.table_name not in table_names:
                continue

            if _table_meta.description:
                tables_info += f"Information of Table {_table_meta.table_name}: {_table_meta.description}\n"
            else:
                tables_info += f"Information of Table {_table_meta.table_name}:\n"

            for _column in _table_meta.columns:
                # Extract column name and comment from the column metadata
                if _column.comment:
                    tables_info += f"Column name: {_column.column_name}, description: {_column.comment}, \n"
                else:
                    tables_info += f"Column name: {_column.column_name}, \n"

        if not tables_info:
            raise ValueError(
                f"No information found for the given tables. Please check the input tables: {table_names}."
            )

        logger.debug(f"Found information for the given tables: {tables_info}.")

        return tables_info


class TablesSchemaTool(BaseSQLAgentTool, BaseTool):
    """
    Return tables schema for the given table names.
    """

    name = "DatabaseTablesSchema"
    description = """
        Input: A list of table names.
        
        Function: Use this tool to get all columns information for relevant tables and \
        identify those possible relevant columns with user posed question.
        
        Output: Schema of the input tables.
        
        Input Example: ["table1", "table2"]
        """

    def _run(
        self,
        table_names: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        logger.info(f"The Agent is calling tool: {self.name}. Input table names: {table_names}.")

        all_available_table_names = self.db_manager.get_available_table_names()
        if not all_available_table_names:
            raise ValueError("No table names found in the database. Please check the database connection.")

        if not table_names:
            missing_tables = set(all_available_table_names) - set(table_names)
            if missing_tables:
                raise ValueError(f"Table names: {missing_tables} not found in the database. Please check the input.")

        tables_schema = [self.db_manager.get_table_schema(table_name) for table_name in table_names]
        return tables_schema


class SQLAgentToolkits(BaseToolkit):
    """
    Return all available tools that the SQL Agent need.
    """

    db_manager: DBMetadataManager = Field(exclude=True)
    embedding: Union[HuggingFaceEmbeddings, AzureOpenAIEmbeddings] = Field(exclude=True)

    class Config(BaseToolkit.Config):
        """Config for Pydantic BaseModel"""

        arbitrary_types_allowed = True
        extra = "allow"

    def get_tools(self) -> List[BaseTool]:
        # TODO: Add tools choice for the agent
        _tools = []

        # Find relevant tables tool
        relevant_tables_tool = RelevantTablesTool(db_manager=self.db_manager, embedding=self.embedding)
        _tools.append(relevant_tables_tool)

        # Get information for given tables tool
        tables_info_tool = TablesInfoTool(db_manager=self.db_manager)
        _tools.append(tables_info_tool)

        return _tools
