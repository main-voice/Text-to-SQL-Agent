import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from text_to_sql.config.settings import AZURE_API_KEY, AZURE_ENDPOINT, PER_API_KEY, PER_ENDPOINT
from text_to_sql.llm.llm_proxy import AzureLLMConfig, LLMProxy, PerplexityLLMConfig


class TestLLMProxy(unittest.TestCase):
    """
    from text_to_sql.llm.llm_proxy import AzureLLMConfig, LLMProxy, PerplexityLLMConfig
    """

    default_question = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="Hi, how are you?"),
    ]

    def test_azure_llm_proxy(self):
        config = AzureLLMConfig(endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY)
        llm_proxy = LLMProxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

        llm_proxy = LLMProxy(AzureLLMConfig())
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_perplexity_llm_proxy(self):
        config = PerplexityLLMConfig(endpoint=PER_ENDPOINT, api_key=PER_API_KEY)
        llm_proxy = LLMProxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

        llm_proxy = LLMProxy(PerplexityLLMConfig())
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)


if __name__ == "__main__":
    unittest.main()
