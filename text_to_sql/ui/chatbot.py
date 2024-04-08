"""
The entry point for the text-to-SQL chatbot demo.

"""

import gradio as gr

from text_to_sql.llm import EmbeddingProxy, LLMProxy
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class SQLChatbot:
    """
    A chatbot which wrapper sql agent and gradio chatbot
    """

    def __init__(self, db_config=None, title=None, description=None, examples=None):
        self.agent = SQLGeneratorAgent(llm_proxy=LLMProxy(), embedding_proxy=EmbeddingProxy(), db_config=db_config)
        self.bot_title = title or "Text-to-SQL Chatbot Demo"
        self.bot_description = (
            description
            or "For now only one build-in database is supported, \
        you can ask related question to the chatbot."
        )
        self.examples = examples or ["Find the user who posted the most number of posts."]

        self.chatbot = self.create_gradio_chatbot()

    def create_gradio_chatbot(self):
        demo = gr.ChatInterface(
            fn=self.chatbot_response,
            examples=self.examples,
            title=self.bot_title,
            description=self.bot_description,
            theme=gr.themes.Soft(),
        )

        return demo

    def chatbot_response(self, message, history):

        response = self.agent.generate_sql_with_agent(message)
        logger.info(f"Chatbot response: {response}")
        return response
