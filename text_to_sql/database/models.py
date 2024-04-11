"""
This file is to store all pydantic models related to database, not AI models
"""

from typing import Any, List

from pydantic import BaseModel, Field, validator

from text_to_sql.config.settings import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER


class DBConfig(BaseModel):
    """
    Store the configuration info of the database
    Use the info specified in the .env file by default
    """

    db_host: str = DB_HOST
    db_name: str = DB_NAME
    db_password: str = Field(default=DB_PASSWORD, exclude=True)
    db_user: str = DB_USER
    db_type: str = Field(default="mysql")


class ColumnMetadata(BaseModel):
    """
    The class to store the metadata of a column
    """

    column_name: str = Field(alias="name")
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
    description: str | None = None


class DatabaseMetadata(BaseModel):
    """
    The class to store the metadata of a database
    """

    tables: List[TableMetadata]
    description: str | None = None
