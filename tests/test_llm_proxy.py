import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from text_to_sql.llm.llm_config import (
    AzureLLMConfig,
    BaseLLMConfig,
    DeepSeekLLMConfig,
    LLama3LLMConfig,
    PerplexityLLMConfig,
)
from text_to_sql.llm.llm_proxy import LLMProxy


class TestLLMProxy(unittest.TestCase):
    """
    from text_to_sql.llm.llm_proxy import AzureLLMConfig, LLMProxy, PerplexityLLMConfig
    """

    default_question = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="Hi, how are you?"),
    ]

    def test_azure_llm_proxy(self):
        config = AzureLLMConfig()
        llm_proxy = LLMProxy.create_llm_proxy(config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_perplexity_llm_proxy(self):
        config = PerplexityLLMConfig()
        llm_proxy = LLMProxy.create_llm_proxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_llama3_llm_proxy(self):
        config = LLama3LLMConfig()
        llm_proxy = LLMProxy.create_llm_proxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_deepseek_llm_proxy(self):
        config = DeepSeekLLMConfig()
        llm_proxy = LLMProxy.create_llm_proxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_llm_proxy_error(self):

        with self.assertRaises(ValueError):
            LLMProxy.create_llm_proxy(config=BaseLLMConfig())


if __name__ == "__main__":
    unittest.main()
