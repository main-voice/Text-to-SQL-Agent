"""Test UI module."""

import unittest

from text_to_sql.ui.chatbot import SQLChatbot


class TestChatbot(unittest.TestCase):
    """Test chatbot module."""

    def setUp(self) -> None:
        self.chatbot = SQLChatbot()

    def test_chat(self):
        self.chatbot.create_gradio_chatbot()
