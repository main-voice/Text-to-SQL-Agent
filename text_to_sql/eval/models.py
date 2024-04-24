"""Pydantic models for evaluation agent"""
from typing import Optional

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    """Evaluation item class"""

    question: str
    golden_query: str = Field(alias="query")
    db_name: str
    query_category: str
    instructions: Optional[str] = Field(default=None)


class EvalResultItem(BaseModel):
    """Evaluation result item class"""

    question: str
    db_name: str
    query_category: str
    instructions: Optional[str] = Field(default=None)

    golden_query: str = Field(alias="query")
    generated_query: str = Field(default=None)
    is_correct: Optional[bool] = Field(default=None)
    exec_correct: Optional[bool] = Field(default=None)
