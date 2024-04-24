"""
This file is used to obtain the metadata of the database based on SqlAlchemy package
"""

import re
from typing import List

from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.sql.ddl import CreateTable

from text_to_sql.database.db_engine import DBEngine
from text_to_sql.database.models import ColumnMetadata, DatabaseMetadata, TableMetadata
from text_to_sql.utils.logger import get_logger

# Notice, Engine is a class from sqlalchemy, while DBEngine is a class from db_engine.py, which is used to interact
# with the database

logger = get_logger(__name__)


class DBMetadataManager:
    """
    Collect the metadata for a given database (including tables info, columns info, primary keys, foreign keys, indexes)
    """

    def __init__(self, db_engine: DBEngine):
        self.db_engine = db_engine
        self._metadata = MetaData()
        self._inspector = None

        self.reflect_metadata()

    def reflect_metadata(self):
        """
        Reflect the metadata of the database
        """
        engine = create_engine(self.db_engine.get_connection_url())
        self._metadata.reflect(bind=engine)
        self._inspector = inspect(engine)
        logger.info(f"Reflected metadata for database {self.db_engine.db_config.db_name}")

    def get_db_metadata(self) -> DatabaseMetadata:
        tables = self._metadata.tables.keys()
        tables_meta = []

        for table_name in tables:
            table_metadata = self.get_table_metadata(table_name)
            tables_meta.append(table_metadata)

        return DatabaseMetadata(tables=tables_meta)

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        """
        Get metadata for a given table name
        """

        _columns = self._inspector.get_columns(table_name=table_name)
        _columns_meta = []
        for column in _columns:
            _columns_meta.append(self.get_column_metadata(column))

        return TableMetadata(
            table_name=table_name,
            columns=_columns_meta,
        )

    @classmethod
    def get_column_metadata(cls, column: dict) -> ColumnMetadata:
        return ColumnMetadata(**column)

    def get_column_metadata_from_name(self, table_name: str, column_name: str) -> ColumnMetadata:
        """
        Get metadata for a given column name
        """
        columns_meta = self._inspector.get_columns(table_name=table_name)
        for column_meta in columns_meta:
            if column_meta["name"] == column_name:
                return self.get_column_metadata(column_meta)

    def get_available_table_names(self) -> List[str]:
        """
        return the list of table names in the database
        """
        return list(self._metadata.tables.keys())

    def get_table_schema(self, table_name: str) -> str:
        """
        return the schema of the table
        """
        table_meta = self._metadata.tables[table_name]
        _engine = create_engine(self.db_engine.get_connection_url())
        table_ddl = str(CreateTable(table_meta).compile(_engine))
        table_ddl = table_ddl.replace("\n", " ").replace("\t", " ")
        # remove multiple spaces
        table_ddl = re.sub(r" +", " ", table_ddl)
        table_ddl = table_ddl.strip()
        return table_ddl

    def get_tables_schema(self, table_names: list) -> dict:
        """
        return the schema of the tables
        """
        tables_schema = {}
        for table_name in table_names:
            tables_schema[table_name] = self.get_table_schema(table_name)
        return tables_schema

    def get_sample_data_of_table(self, table_name: str, limit: int = 3):
        """
        return the sample data of the table
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.db_engine.execute(query)

    def get_sample_data_of_column(self, table_name: str, column_name: str, limit: int = 3):
        """
        return the sample data of the column
        """
        query = f"SELECT {column_name} FROM {table_name} LIMIT {limit}"
        return self.db_engine.execute(query)
