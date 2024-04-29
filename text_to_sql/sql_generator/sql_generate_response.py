"""Wrapper class for SQL agent response"""


class SQLGeneratorResponse:
    """Wrapper the sql generation response from the SQL agent with additional information"""

    def __init__(
        self,
        question: str,
        db_name: str,
        generated_sql: str,
        llm_source: str,
        llm_model: str,
        token_usage: int = -1,
        error=None,
    ):
        self.question = question
        self.db_name = db_name
        self.generated_sql = generated_sql
        self.llm_source = llm_source
        self.llm_model = llm_model
        self.token_usage = token_usage
        self.error = error or None

    def __str__(self):
        if self.token_usage > 0:
            return f"Question: {self.question}\n Database: {self.db_name}\n SQL: {self.generated_sql}\n \
            Token Usage: {self.token_usage}"

        return f"Question: {self.question}\n Database: {self.db_name}\n SQL: {self.generated_sql}"
