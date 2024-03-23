from pydantic import BaseModel
from typing import Literal, Optional, Any
from langchain_community.chat_models.azure_openai import AzureChatOpenAI

from text_to_sql.config.settings import AZURE_API_KEY, AZURE_ENDPOINT


class BaseLLMConfig(BaseModel):
    """
    The Base LLM configurations class
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    model: str


class AzureLLMConfig(BaseLLMConfig):
    """
    Azure LLM configuration class
    """

    azure_endpoint: Optional[str] = None
    deployment_name: Literal["gpt-4", "gpt-35-turbo", "gpt-4-turbo"] = "gpt-35-turbo"
    model: Literal["gpt-4", "gpt-35-turbo"] = "gpt-35-turbo"
    api_version: str = "2023-08-01-preview"
    api_key: Optional[str] = None


class LLMProxy:

    @property
    def azure_llm(self):
        llm_config = AzureLLMConfig(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY)

        llm = AzureChatOpenAI(
            azure_endpoint=llm_config.azure_endpoint,
            deployment_name=llm_config.deployment_name,
            model=llm_config.model,
            openai_api_type="azure",
            openai_api_version=llm_config.api_version,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            openai_api_key=llm_config.api_key,
        )
        return llm

    def get_response_from_llm(self, question: Any, llm_name: str = "azure"):
        if llm_name == "azure":
            return self.azure_llm.invoke(question)
        else:
            raise ValueError("Only Azure LLM supported now!")
