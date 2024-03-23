import unittest

from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent


class TestSQLGeneratorAgent(unittest.TestCase):
    def setUp(self):
        self.sql_generator_agent = SQLGeneratorAgent(db_metadata_manager=DBMetadataManager(MySQLEngine()),
                                                     llm_proxy=LLMProxy())

    def test_generate_sql(self):
        sql = self.sql_generator_agent.generate_sql("Show me the names of all the users")
        self.assertEqual(sql, "select username from jk_user;")

        sql = self.sql_generator_agent.generate_sql("Show me the users whose name is 'test'")
        self.assertEqual(sql, "select * from jk_user where username = 'test';")
