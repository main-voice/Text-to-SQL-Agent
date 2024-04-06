import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from text_to_sql.llm import LLMProxy


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
        Test openai llm, should accept a ValueError as haven't integrated openai llm
        """
        with self.assertRaises(ValueError):
            llm_proxy = LLMProxy(llm_name="openai")
            llm_proxy.get_response_from_llm(self.default_question)


if __name__ == "__main__":
    unittest.main()
