"""
Tools for SQL Generate Agent
Tools are interfaces that an agent can use to interact with the world.

For details: ref to "https://python.langchain.com/docs/modules/agents/tools/"
"""

# TODO: Add a Chinese to English translation tool

from typing import List, Any, Optional

import numpy as np
import pandas as pd
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from text_to_sql.database.models import TableMetadata
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class RelevantTablesTool(BaseTool):
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
    tables_context: List[TableMetadata]
    embedding: HuggingFaceEmbeddings
    top_k: int = 5  # The number of most similar tables to return, default is 5

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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """
        Find all possible relevant tables and columns in the database based on user question and db metadata.
        """
        logger.info(f"The Agent is calling tool: {self.name}.")

        if not self.tables_context:
            raise ValueError("No table metadata found in the database. Please check the database connection.")

        question_embedding = self.generate_embedding(question)

        # Contain all tables and columns string representation
        tables_cols_str = []

        # Convert table metadata to a string
        for _table_meta in self.tables_context:
            _columns_str = ""
            for _column in _table_meta.columns:
                # Extract column name and comment from the column metadata to generate embedding
                if _column.comment:
                    _columns_str += f"{_column.column_name}: {_column.comment}, "
                else:
                    _columns_str += f"{_column.column_name}, "

            _table_str = f"Table {_table_meta.table_name} contain columns: [{_columns_str}]."
            if _table_meta.description:
                _table_str += f" and table description: {_table_meta.description}"

            tables_cols_str.append([_table_meta.table_name, _table_str])

        df = pd.DataFrame(tables_cols_str, columns=["table_name", "table_col_str"])
        df["embedding"] = self.generate_doc_embedding(df["table_col_str"].tolist())
        df["similarity"] = df["embedding"].apply(lambda x: self.cosine_similarity(question_embedding, x))

        df.sort_values(by="similarity", ascending=False, inplace=True)
        df = df.head(self.top_k)

        if df.empty:
            logger.error("No relevant tables found in the database. Please check the input question.")
            return []

        table_similarity_score = df[["table_name", "similarity"]].values.tolist()
        logger.info(
            f"Found relevant tables and relevance scores: {table_similarity_score} for the question: {question}."
        )

        # Now we have the table names and their relevance scores, return the most relevant tables
        return df[["table_name"]].values.tolist()


class InfoTablesTool(BaseTool):
    """
    Return information for given table names.
    """

    name = "DatabaseTablesInformation"
    description = """
        Input: A list of table names.
        Function: Use this tool to get metadata information for a list of tables in the database.
        Output: Metadata for a list of tables.
        
        Input Example: ["table1", "table2"]
        """
    tables_context: List[TableMetadata]

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


class SQLAgentToolkits(BaseToolkit):
    """
    Return all available tools that the SQL Agent need.
    """

    tables_context: List[TableMetadata]
    embedding: HuggingFaceEmbeddings

    def get_tools(self) -> List[BaseTool]:
        # TODO: Add tools choice for the agent
        _tools = []

        # Find relevant tables tool
        relevant_tables_tool = RelevantTablesTool(tables_context=self.tables_context, embedding=self.embedding)
        _tools.append(relevant_tables_tool)

        # Get information for given tables tool
        info_tables_tool = InfoTablesTool(tables_context=self.tables_context)
        _tools.append(info_tables_tool)

        return _tools
