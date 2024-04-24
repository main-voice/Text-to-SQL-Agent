import unittest

from langchain_core.messages import HumanMessage, SystemMessage

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
        config = AzureLLMConfig()
        llm_proxy = LLMProxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_perplexity_llm_proxy(self):
        config = PerplexityLLMConfig()
        llm_proxy = LLMProxy(config=config)
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None


if __name__ == "__main__":
    unittest.main()
