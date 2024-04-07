"""
This file is to store all pydantic models related to database, not AI models
"""

from typing import Any, List

from pydantic import BaseModel, Field, validator

from text_to_sql.database.db_engine import DBEngine


class ColumnMetadata(BaseModel):
    """
    The class to store the metadata of a column
    """

    column_name: str = Field(alias="name")
    is_nullable: bool = Field(alias="nullable")
    column_default: str | None = Field(alias="default")
    comment: str | None
    type: Any = "str"

    @validator("type")
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
    description: str | None


class DatabaseMetadata(BaseModel):
    """
    The class to store the metadata of a database
    """

    db_engine: DBEngine
    tables: List[TableMetadata]
    description: str | None

    class Config:
        """
        pydantic config class, to allow custom class DBEngine to be used
        """

        arbitrary_types_allowed = True
