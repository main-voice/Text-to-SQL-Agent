"""
This file is used to obtain the metadata of the database based on SqlAlchemy package
"""

from sqlalchemy import create_engine, MetaData, inspect

from text_to_sql.utils.logger import get_logger
from .db_engine import DBEngine
from .models import ColumnMetadata, TableMetadata, DatabaseMetadata

# Notice, Engine is a class from sqlalchemy, while DBEngine is a class from db_engine.py, which is used to interact
# with the database

logger = get_logger(__name__)


class DBMetadataManager:
    """
    Collect the metadata for a given database (including tables info, columns info, primary keys, foreign keys, indexes)
    """

    sql_meta: MetaData
    inspector: inspect
    db_engine: DBEngine

    def __init__(self, db_engine: DBEngine):
        self.db_engine = db_engine
        self.create_inspector()

    def create_inspector(self):
        logger.debug("Creating inspector using connection url...")

        sqlalchemy_engine = create_engine(self.db_engine.get_connection_url())
        self.sql_meta = MetaData()
        self.sql_meta.reflect(bind=sqlalchemy_engine)
        self.inspector = inspect(sqlalchemy_engine)

        logger.debug("Create inspector successfully.")
        return self.inspector

    def get_db_metadata(self) -> DatabaseMetadata:
        tables = self.inspector.get_table_names()
        tables_meta = []

        for table_name in tables:
            table_metadata = self.get_table_metadata(table_name)
            tables_meta.append(table_metadata)

        return DatabaseMetadata(db_engine=self.db_engine, tables=tables_meta)

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        """
        Get metadata for a given table name
        """

        _columns = self.inspector.get_columns(table_name=table_name)
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
        return self.get_column_metadata(self.inspector.get_columns(table_name=table_name, column_name=column_name)[0])
