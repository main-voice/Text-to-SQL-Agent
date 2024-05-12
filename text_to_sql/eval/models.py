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

    error_detail: str = Field(default=None, description="Error detail message if any")


class BoxPlotItem(BaseModel):
    """Item for box plot"""

    min_value: float
    max_value: float
    avg_value: float
    median: float  # 50th percentile, q2
    q1: float
    q3: float
    outliers: Optional[list] = Field(default=[])


class AnalyzerItem(BaseModel):
    """Item for analyzer and visualizer"""

    platform: str = Field(description="The LLM platform")
    llm_model: str = Field(description="The LLM model")
    method: str = Field(description="The method used for the evaluation")
    accuracy: float = Field(description="The accuracy rate of the evaluation result")
    token_usage: Optional[BoxPlotItem] = Field(
        description="The average token usage of the evaluation result", default=None
    )

    eval_duration: Optional[BoxPlotItem] = Field(
        description="The average evaluation duration of the evaluation result", default=None
    )
