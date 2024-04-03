from text_to_sql.database.db_config import DBConfig
from text_to_sql.database.db_engine import MySQLEngine
from text_to_sql.database.db_metadata_manager import DBMetadataManager
from text_to_sql.llm import LLMProxy, EmbeddingProxy
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent

if __name__ == "__main__":
    sql_agent = SQLGeneratorAgent(
        db_metadata_manager=DBMetadataManager(MySQLEngine(DBConfig())),
        llm_proxy=LLMProxy(),
        embedding_proxy=EmbeddingProxy(embedding_source="huggingface")
    )

    question = "Find the user who has the most posts"
    question_cn = "找到发帖数量最多的用户"
    result = sql_agent.generate_sql_with_agent(question, verbose=True)
    print(result)
