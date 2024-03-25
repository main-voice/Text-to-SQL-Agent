import unittest
from text_to_sql.llm.llm_proxy import LLMProxy
from langchain_core.messages import HumanMessage, SystemMessage


class TestLLMProxy(unittest.TestCase):
    """
    Class to test LLM Proxy
    """

    default_question = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="Hi, how are you?"),
    ]

    def test_azure_llm_proxy(self):
        llm_proxy = LLMProxy()
        response = llm_proxy.get_response_from_llm(self.default_question, verbose=True)
        print(response)
        assert response is not None

    def test_openai_llm_proxy(self):
        """
        Test openai llm, should accept a ValueError as haven't integrate openai llm
        """
        with self.assertRaises(ValueError):
            llm_proxy = LLMProxy()
            response = llm_proxy.get_response_from_llm(self.default_question, llm_name="openai")


if __name__ == "__main__":
    unittest.main()
