import os
import unittest

from dotenv import load_dotenv

from text_to_sql.llm.llm_config import AzureLLMConfig, LLama3LLMConfig, PerplexityLLMConfig


class TestLLMConfig(unittest.TestCase):
    """Test the LLMConfig class, mainly to check if the class read the environment variables correctly"""

    def setUp(self):
        """
        set up config
        """
        self.test_env_path = ".test.env"
        self.test_env_abs_path = os.path.join(os.path.dirname(__file__), self.test_env_path)

        load_dotenv(dotenv_path=self.test_env_abs_path)

    def test_load_llm_config(self):
        """
        Test if pydantic can read the environment variables automatically and map them to the class
        """
        azure_llm_config = AzureLLMConfig(_env_file=self.test_env_abs_path)
        assert azure_llm_config.endpoint == os.environ.get("AZURE_ENDPOINT")
        assert azure_llm_config.api_key.get_secret_value() == os.getenv("AZURE_API_KEY")

        print(azure_llm_config.dict())

        perplexity_llm_config = PerplexityLLMConfig(_env_file=self.test_env_abs_path)
        assert perplexity_llm_config.endpoint == os.environ.get("PERPLEXITY_ENDPOINT")
        assert perplexity_llm_config.api_key.get_secret_value() == os.getenv("PERPLEXITY_API_KEY")
        print(perplexity_llm_config.dict())

        llama3_llm_config = LLama3LLMConfig(_env_file=self.test_env_abs_path)
        assert llama3_llm_config.endpoint == os.environ.get("LLAMA3_ENDPOINT")
        assert llama3_llm_config.api_key.get_secret_value() == os.getenv("LLAMA3_API_KEY")
        print(llama3_llm_config.dict())
