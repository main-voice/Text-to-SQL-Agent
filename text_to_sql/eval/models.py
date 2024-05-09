"""Pydantic models for evaluation agent"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    """Evaluation item class, the class is mapped to the evaluation dataset directly"""

    question: str
    golden_query: str = Field(alias="query")
    db_name: str
    query_category: str
    instructions: Optional[str] = Field(default=None)


class SQLHardness(str, Enum):
    """Hardness level for a SQL statement"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ULTRA = "ultra"


class EvalResultItem(BaseModel):
    """Evaluation result item class"""

    question: str
    db_name: str
    query_category: str
    instructions: Optional[str] = Field(default=None)

    golden_query: str
    hardness: SQLHardness = Field(default=None)
    generated_query: str = Field(default=None)

    token_usage: int = Field(default=-1)
    is_correct: Optional[bool] = Field(default=False)
    exec_correct: Optional[bool] = Field(default=False)
    eval_duration: Optional[float] = Field(default=None)
