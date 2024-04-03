from .embedding_proxy import EmbeddingProxy
from .llm_proxy import LLMProxy

"""
The two proxy class should be the only classes that are exposed to the outside world
"""

__all__ = [
    EmbeddingProxy,
    LLMProxy
]
