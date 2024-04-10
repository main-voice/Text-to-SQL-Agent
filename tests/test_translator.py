import unittest

from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.translator import LLMTranslator, YoudaoTranslator

logger = get_logger(__name__)


class TestTranslator(unittest.TestCase):
    def setUp(self) -> None:
        self.default_question = "你好，世界！"
        self.default_answer = ["hello, world", "hello world", "helloworld", "hello,world"]

    def test_llm_translator(self):
        self.translator = LLMTranslator()
        resp = self.translator.translate(to_be_translate=self.default_question, verbose=True)

        print(f"The translation of {self.default_question} from {self.translator.translate_source} is {resp}")
        self.assertTrue(any(answer in resp.lower() for answer in self.default_answer))

    def test_youdao_translator(self):
        self.translator = YoudaoTranslator()
        resp = self.translator.translate(to_be_translate=self.default_question)

        print(f"The translation of {self.default_question} from {self.translator.translate_source} is {resp}")
        self.assertTrue(any(answer in resp.lower() for answer in self.default_answer))


if __name__ == "__main__":
    unittest.main()
