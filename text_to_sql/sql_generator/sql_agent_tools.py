"""
Tools for SQL Generate Agent
Tools are interfaces that an agent can use to interact with the world.

For details: ref to "https://python.langchain.com/docs/modules/agents/tools/"
"""

from typing import List, Any, Optional

import numpy as np
import pandas as pd
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool

from text_to_sql.database.models import TableMetadata
from text_to_sql.llm.llm_proxy import get_huggingface_embedding


class RelevantTablesSQLAgent(BaseTool):
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
    tables_context: List[TableMetadata] = []
    embedding = get_huggingface_embedding()
    # Number of tables to be returned
    top_k: int = 5

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
        df["similarity"] = df["embedding"].apply(
            lambda x: self.cosine_similarity(question_embedding, x)
        )

        df.sort_values(by="similarity", ascending=False, inplace=True)
        df = df.head(self.top_k)

        # Now we have the table names and their relevance scores, return the most relevant tables
        return df[["table_name"]].values.tolist()

