import os
import unittest

from text_to_sql.llm.llm_config import (
    AzureLLMConfig,
    DeepSeekLLMConfig,
    LLama3LLMConfig,
    PerplexityLLMConfig,
    ZhiPuLLMConfig,
)


class TestLLMConfig(unittest.TestCase):
    """Test the LLMConfig class, mainly to check if the class read the environment variables correctly"""

    def setUp(self):
        """
        set up config
        """
        self.test_env_path = ".test.env.text"
        self.test_env_abs_path = os.path.join(os.path.dirname(__file__), self.test_env_path)

        # load_dotenv(dotenv_path=self.test_env_abs_path)

    def test_load_llm_config(self):
        """
        Test if pydantic can read the environment variables automatically and map them to the class
        """
        azure_llm_config = AzureLLMConfig(_env_file=self.test_env_abs_path)
        assert azure_llm_config.endpoint is not None
        assert azure_llm_config.api_key.get_secret_value() is not None

        print(azure_llm_config.dict())

        perplexity_llm_config = PerplexityLLMConfig(_env_file=self.test_env_abs_path)
        assert perplexity_llm_config.endpoint is not None
        assert perplexity_llm_config.api_key.get_secret_value() is not None
        print(perplexity_llm_config.dict())

        llama3_llm_config = LLama3LLMConfig(_env_file=self.test_env_abs_path)
        assert llama3_llm_config.endpoint is not None
        assert llama3_llm_config.api_key.get_secret_value() is not None
        print(llama3_llm_config.dict())

        deepseek_llm_config = DeepSeekLLMConfig(_env_file=self.test_env_abs_path)
        assert deepseek_llm_config.endpoint is not None
        assert deepseek_llm_config.api_key.get_secret_value() is not None
        print(deepseek_llm_config.dict())

        zhipu_llm_config = ZhiPuLLMConfig(_env_file=self.test_env_abs_path)
        assert zhipu_llm_config.endpoint is not None
        assert zhipu_llm_config.api_key.get_secret_value() is not None
        print(zhipu_llm_config.dict())
