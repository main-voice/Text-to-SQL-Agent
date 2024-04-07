import unittest
from typing import List

from text_to_sql.llm import EmbeddingProxy


class TestEmbedding(unittest.TestCase):
    """
    Class to test LLM Embedding
    """

    default_str = "string to test embedding"

    def test_azure_embedding_with_name(self):
        azure_embedding = EmbeddingProxy(embedding_source="azure", model_name="Text-Embedding-Ada-002").get_embedding()
        result = azure_embedding.embed_query(self.default_str)
        print(result)
        self.assertTrue(isinstance(result, List))

    def test_azure_embedding_without_name(self):
        azure_embedding = EmbeddingProxy(embedding_source="azure").get_embedding()
        result = azure_embedding.embed_query(self.default_str)
        print(result)
        self.assertTrue(isinstance(result, List))

    def test_wrong_embedding_model_name(self):
        test_model_name = "non-existing"
        with self.assertRaises(ValueError):
            EmbeddingProxy(embedding_source="azure", model_name=test_model_name).get_embedding()

        with self.assertRaises(ValueError):
            EmbeddingProxy(embedding_source="huggingface", model_name=test_model_name).get_embedding()

    def test_huggingface_embedding_with_name(self):
        embedding = EmbeddingProxy(
            embedding_source="huggingface", model_name="sentence-transformers/all-mpnet-base-v2"
        ).get_embedding()
        result = embedding.embed_query(self.default_str)
        print(result)
        self.assertTrue(isinstance(result, List))

    def test_huggingface_embedding_without_name(self):
        embedding = EmbeddingProxy(embedding_source="huggingface").get_embedding()
        result = embedding.embed_query(self.default_str)
        print(result)
        self.assertTrue(isinstance(result, List))
