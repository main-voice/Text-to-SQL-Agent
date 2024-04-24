"""
Proxy class to supported LLM client
"""

from typing import Any

# pylint: disable=no-name-in-module
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from text_to_sql.utils.logger import get_logger

from .llm_config import AzureLLMConfig, LLama3LLMConfig, PerplexityLLMConfig

logger = get_logger(__name__)


class AzureLLM:
    """The Azure LLM class"""

    def __init__(self, config: AzureLLMConfig):
        self.llm = self.get_llm(config)

    @staticmethod
    def get_llm(config: AzureLLMConfig) -> AzureChatOpenAI:
        """Static method, will return a Azure LLM instance according to the configuration provided

        Args:
            config (AzureLLMConfig): The config information for the Azure LLM

        Returns:
            AzureChatOpenAI: An azure llm instance defined in langchain_openai
        """
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


class PerplexityLLM:
    """The Perplexity LLM class"""

    def __init__(self, config: PerplexityLLMConfig):
        self.llm = self.get_llm(config)

    @staticmethod
    def get_llm(config: PerplexityLLMConfig) -> ChatOpenAI:
        """Static method, will return a Perplexity LLM instance according to the configuration provided

        Args:
            config (PerplexityLLMConfig): The config information for the Perplexity LLM

        Returns:
            ChatOpenAI: A perplexity llm instance defined in langchain_openai
        """
        return ChatOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class LLama3LLM:
    """The LLama3 LLM class"""

    def __init__(self, config: LLama3LLMConfig):
        self.llm = self.get_llm(config)

    @staticmethod
    def get_llm(config: LLama3LLMConfig) -> ChatOpenAI:
        """Static method, will return a LLama3 LLM instance according to the configuration provided

        Args:
            config (LLama3LLMConfig): The config information for the LLama3 LLM

        Returns:
            ChatOpenAI: A llama3 llm instance defined in langchain_openai
        """
        return ChatOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            model=config.llm_source + "/" + config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class LLMProxy:
    """
    A Proxy class to interact with the supported LLMs
    """

    def __init__(self, config: AzureLLMConfig | PerplexityLLMConfig | LLama3LLMConfig) -> None:
        if config.llm_source == "azure":
            self.llm = AzureLLM(config).llm
        elif config.llm_source == "perplexity":
            self.llm = PerplexityLLM(config).llm
        elif config.llm_source == "meta":
            self.llm = LLama3LLM(config).llm
        else:
            raise ValueError("Only [Azure, Perplexity, LLama3] LLM supported now!")

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
