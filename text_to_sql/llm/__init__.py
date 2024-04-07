"""
The two proxy class should be the only classes that are exposed to the outside world
"""

from .embedding_proxy import EmbeddingProxy
from .llm_proxy import LLMProxy

__all__ = ["EmbeddingProxy", "LLMProxy"]
