"""
This file is used to obtain the metadata of the database based on SqlAlchemy package
"""

from sqlalchemy import create_engine, MetaData, inspect, Engine
from .db_config import DBConfig
from .db_engine import MySQLEngine, DBEngine

# Notice, Engine is a class from sqlalchemy, while DBEngine is a class from db_engine.py, which is used to interact
# with the database


class MetadataCollector:
    """
    Collect the metadata for a given database
    """

    meta: dict = {}
    inspector: inspect = None
    db_engine: DBEngine

    def __init__(self, db_engine: DBEngine):
        self.db_engine = db_engine
        self.create_inspector()

    def create_inspector(self):
        sqlalchemy_engine = create_engine(self.db_engine.get_connection_url())
        meta = MetaData()
        meta.reflect(bind=sqlalchemy_engine)
        self.inspector = inspect(sqlalchemy_engine)
        return self.inspector

    def get_db_metadata(self):
        tables = self.inspector.get_table_names()

        for table_name in tables:
            self.meta[table_name] = {
                "columns": self.inspector.get_columns(table_name),
                "primary_keys": self.inspector.get_pk_constraint(table_name),
                "foreign_keys": self.inspector.get_foreign_keys(table_name),
                "indexes": self.inspector.get_indexes(table_name),
            }

        return self.meta

    def get_table_metadata(self, table_name: str):
        if self.meta:
            return self.meta[table_name]

        tables = self.inspector.get_table_names()
        if table_name in tables:
            return {
                "columns": self.inspector.get_columns(table_name),
                "primary_keys": self.inspector.get_pk_constraint(table_name),
                "foreign_keys": self.inspector.get_foreign_keys(table_name),
                "indexes": self.inspector.get_indexes(table_name),
            }

    def get_column_metadata(self, table_name: str, column_name: str):
        columns = self.inspector.get_columns(table_name)
        for column in columns:
            if column["name"] == column_name:
                return column

        return None
