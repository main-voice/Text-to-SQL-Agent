# Example of using the SQLGeneratorAgent to generate SQL from a question

from text_to_sql.ui.chatbot import SQLChatbot


if __name__ == "__main__":
    examples = [
        "Find the user who has the most posts",
        "找到发帖数量最多的用户",
        "Show me the users who login in today"
    ]

    sql_chatbot = SQLChatbot(examples=examples)

    demo = sql_chatbot.create_gradio_chatbot()
    demo.queue().launch(
        server_name="localhost",
        server_port=7860
    )
