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

        return DatabaseMetadata(db_engine=self.db_engine, tables=tables_meta)

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
        return self.get_column_metadata(self._inspector.get_columns(table_name=table_name, column_name=column_name)[0])
