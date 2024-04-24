"""
This file is to store all pydantic models related to database, not AI models
"""

from typing import Any, List

from pydantic import BaseModel, Field, validator


class ColumnMetadata(BaseModel):
    """
    The class to store the metadata of a column
    """

    column_name: str = Field(alias="name")
    column_default: str | None = Field(alias="default")
    comment: str | None = Field(alias="column_description")
    data_type: Any = Field(alias="type")

    @validator("data_type")
    @classmethod
    def check_type(cls, input_type):
        # A demo input type is 'type': VARCHAR(charset='utf8mb4', collation='utf8mb4_unicode_ci', length=64)',
        # we transform it to str directly for now.
        return str(input_type)

    @validator("comment")
    @classmethod
    def check_comment(cls, input_comment):
        if input_comment is None:
            return None
        assert isinstance(input_comment, str), "Comment must be a string"

        return input_comment


class TableMetadata(BaseModel):
    """
    The class to store the metadata of a table
    """

    table_name: str
    columns: List[ColumnMetadata]
    description: str | None = None


class DatabaseMetadata(BaseModel):
    """
    The class to store the metadata of a database
    """

    tables: List[TableMetadata]
    description: str | None = None
