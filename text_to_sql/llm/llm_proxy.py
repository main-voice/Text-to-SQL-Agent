"""
Proxy class to supported LLM client
"""

from abc import ABC, abstractmethod
from typing import Any, Type

# pylint: disable=no-name-in-module
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from text_to_sql.utils.logger import get_logger

from .llm_config import AzureLLMConfig, BaseLLMConfig, DeepSeekLLMConfig, LLama3LLMConfig, PerplexityLLMConfig

logger = get_logger(__name__)


class BaseLLM(ABC):
    """The Base LLM class, claims the method that should be implemented by the child classes"""

    @abstractmethod
    def get_llm_instance(self, config: BaseLLMConfig) -> ChatOpenAI:
        """
        Return a LLM instance according to the configuration provided
        """
        pass


class AzureLLM(BaseLLM):
    """The Azure LLM class"""

    def get_llm_instance(self, config: BaseLLMConfig) -> AzureChatOpenAI:
        """Create a Azure LLM instance according to the configuration provided

        Args:
            config (AzureLLMConfig): The config information for the Azure LLM

        Returns:
            AzureChatOpenAI: An azure llm instance defined in langchain_openai
        """
        if not isinstance(config, AzureLLMConfig):
            raise TypeError(f"Expected AzureLLMConfig, got {type(config)}")

        try:
            return AzureChatOpenAI(
                azure_endpoint=config.endpoint,
                deployment_name=config.deployment_name,
                model=config.model,
                openai_api_type=config.llm_source,
                openai_api_version=config.api_version,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=config.api_key,
            )
        except Exception as e:
            logger.error(f"Failed to creat Azure LLM instance: {e}")
            raise e


class PerplexityLLM(BaseLLM):
    """The Perplexity LLM class"""

    def get_llm_instance(self, config: BaseLLMConfig) -> ChatOpenAI:
        """Return a Perplexity LLM instance according to the configuration provided

        Args:
            config (PerplexityLLMConfig): The config information for the Perplexity LLM

        Returns:
            ChatOpenAI: A perplexity llm instance defined in langchain_openai
        """
        if not isinstance(config, PerplexityLLMConfig):
            raise TypeError(f"Expected PerplexityLLMConfig, got {type(config)}")

        try:
            return ChatOpenAI(
                base_url=config.endpoint,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to creat Perplexity LLM instance: {e}")
            raise e


class LLama3LLM(BaseLLM):
    """The LLama3 LLM class"""

    def get_llm_instance(self, config: BaseLLMConfig) -> ChatOpenAI:
        """Static method, will return a LLama3 LLM instance according to the configuration provided

        Args:
            config (LLama3LLMConfig): The config information for the LLama3 LLM

        Returns:
            ChatOpenAI: A llama3 llm instance defined in langchain_openai
        """
        if not isinstance(config, LLama3LLMConfig):
            raise TypeError(f"Expected LLama3LLMConfig, got {type(config)}")

        try:
            return ChatOpenAI(
                base_url=config.endpoint,
                api_key=config.api_key,
                model=config.llm_source + "/" + config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to creat LLama3 LLM instance: {e}")
            raise e


class DeepSeekLLM(BaseLLM):
    """The DeepSeek LLM class"""

    def get_llm_instance(self, config: BaseLLMConfig) -> ChatOpenAI:
        """Static method, will return a DeepSeek LLM instance according to the configuration provided

        Args:
            config (DeepSeekLLMConfig): The config information for the DeepSeek LLM

        Returns:
            ChatOpenAI: A DeepSeek llm instance defined in langchain_openai
        """
        if not isinstance(config, DeepSeekLLMConfig):
            raise TypeError(f"Expected LLama3LLMConfig, got {type(config)}")

        try:
            return ChatOpenAI(
                base_url=config.endpoint,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to creat DeepSeek LLM instance: {e}")
            raise e


class LLMProxy:
    """
    A Proxy class to interact with the supported LLMs
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    @staticmethod
    def create_llm_proxy(config: BaseLLMConfig) -> "LLMProxy":
        """
        Factory method to create a LLMProxy instance
        """
        llm_class: Type[BaseLLM] = {
            "azure": AzureLLM,
            "perplexity": PerplexityLLM,
            "meta": LLama3LLM,
            "deepseek": DeepSeekLLM,
        }.get(config.llm_source, None)

        if llm_class is None:
            raise ValueError(
                f"Unsupported LLM source: {config.llm_source}, only [Azure, Perplexity, LLama3] supported now!"
            )

        llm_instance = llm_class().get_llm_instance(config)

        # return the LLMProxy instance, will call the init method here, and pass the llm_instance
        return LLMProxy(llm_instance)

    def get_response_from_llm(self, question: Any, verbose: bool = False):
        """method to get response from LLM

        Args:
            question (Any): the question to ask the LLM
            verbose (bool, optional): if to print token usage. Defaults to False.

        Returns:
            BaseMessage: An message object with the response from the LLM
        """
        if verbose:
            with get_openai_callback() as cb:
                _response = self.llm.invoke(question)
                logger.info(f"Token usage: {cb.total_tokens}")
                return _response

        return self.llm.invoke(question)
