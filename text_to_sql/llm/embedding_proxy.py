"""
Proxy class to supported embedding models
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, validator
from sentence_transformers import SentenceTransformer

from text_to_sql.config.settings import AZURE_API_KEY, AZURE_ENDPOINT
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedding(BaseModel):
    """
    Base class for embeddings
    """

    model_name: str
    _available_models: List[str] = []
    embedding: Optional[Union[AzureOpenAIEmbeddings, HuggingFaceEmbeddings]] = None

    def get_embedding_model(self) -> Any:
        """
        Get the embedding model
        """
        raise NotImplementedError


class AzureEmbedding(BaseEmbedding, BaseModel):
    """
    Azure OpenAI Embedding
    """

    _available_models: List[str] = ["Text-Embedding-Ada-002", "Text-Embedding-3-Large", "Text-Embedding-3-Small"]

    # NOTICE: Pydantic will convert all function with validator decorator to classmethod
    # If we declare the function as a classmethod, @classmethod must be placed before @validator, or it will not work
    # So here we use @validator("model_name") instead of @classmethod @validator("model_name")
    @validator("model_name")
    def check_model_name(cls, model_name: str) -> str:  # pylint: disable=no-self-argument
        x = type(cls._available_models)
        print(x)
        if model_name not in cls._available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {cls._available_models}")
        return model_name

    def get_embedding_model(self) -> AzureOpenAIEmbeddings:
        """
        Get a AzureOpenAIEmbeddings instance, shouldn't be called directly
        """
        logger.info("Loading Azure OpenAI Embedding model...")

        if self.embedding:
            return self.embedding

        if self.model_name.lower() == "text-embedding-ada-002":
            deploy_name = "embedding"
        else:
            deploy_name = self.model_name.lower()

        logger.debug(f"You are using model: {self.model_name}, the deployment name is: {deploy_name}")

        _azure_embedding = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            deployment=deploy_name,
            model=self.model_name,
            openai_api_key=AZURE_API_KEY,
            openai_api_version="2023-08-01-preview",
        )
        self.embedding = _azure_embedding
        return self.embedding


class HuggingFaceEmbedding(BaseEmbedding, BaseModel):
    """
    Hugging Face Embedding
    """

    _available_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "sentence-transformers/all-MiniLM-L12-v2",
    ]

    @validator("model_name")
    def check_model_name(cls, model_name: str) -> str:  # pylint: disable=no-self-argument
        if model_name not in cls._available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {cls._available_models}")
        return model_name

    def get_embedding_model(self) -> HuggingFaceEmbeddings | SentenceTransformer:
        """
        Get a Hugging Face model
        """
        logger.info("Loading HuggingFace Embedding model...")
        if self.embedding:
            return self.embedding

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {device}")
        cache_folder = str(Path(__file__).resolve().parent / "model_assets")

        model_kwargs = {"device": device}
        sentence_embedding = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
        )
        return sentence_embedding


class EmbeddingProxy:
    """
    A Proxy class to interact with the Embedding classes
    """

    default_embedding_model = {
        "azure": "Text-Embedding-Ada-002",
        "huggingface": "sentence-transformers/all-mpnet-base-v2",
    }

    def __init__(self, embedding_source: str = "huggingface", model_name: Optional[str] = None):
        """
        init an embedding proxy instance using source and model name
        """
        self.embedding_source: str = embedding_source
        self.model_name: Optional[str] = model_name or self.default_embedding_model.get(embedding_source)

        self.embedding: Optional[Union[AzureOpenAIEmbeddings, HuggingFaceEmbeddings]] = None

    def get_embedding(self) -> Union[AzureOpenAIEmbeddings, HuggingFaceEmbeddings]:
        """
        Get the embedding model
        """
        if self.embedding:
            return self.embedding

        if self.embedding_source == "azure":
            logger.info("Choose Azure OpenAI Embedding as the embedding source.")
            self.embedding = AzureEmbedding(model_name=self.model_name).get_embedding_model()
        elif self.embedding_source == "huggingface":
            logger.info("Choose HuggingFace Embedding as the embedding source.")
            self.embedding = HuggingFaceEmbedding(model_name=self.model_name).get_embedding_model()
        else:
            raise ValueError(
                f"Embedding source {self.embedding_source} is not available. "
                f"Available sources are: 'azure' and 'huggingface'"
            )

        return self.embedding
