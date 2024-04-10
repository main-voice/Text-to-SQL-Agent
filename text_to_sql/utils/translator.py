"""
A utils module that translates the input text to English.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Literal

import requests
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

from text_to_sql.config.settings import YD_APP_ID, YD_APP_SECRET_KEY
from text_to_sql.llm.llm_proxy import LLMProxy
from text_to_sql.utils.logger import get_logger
from text_to_sql.utils.prompt import TRANSLATOR_PROMPT

logger = get_logger(__name__)


class BaseTranslator(ABC):
    """Abstract class for translator"""

    def __init__(self, *args, **kwargs):
        self.translate_source: Literal["llm", "youdao"] = "youdao"

    @abstractmethod
    def translate(self, to_be_translate: str, *args, **kwargs) -> str:
        """
        The method to translate the input text to english

        Parameters:
            to_be_translate: the string to be translated

        Returns:
            The translated text (english)
        """
        raise NotImplementedError


class LLMTranslator(BaseTranslator):
    """A translator that uses LLM to translate the input text to English."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translate_source = "llm"
        self.llm_proxy = LLMProxy()

    def translate(self, to_be_translate: str, *args, **kwargs) -> str:
        llm = self.llm_proxy.llm
        prompt_template = PromptTemplate.from_template(TRANSLATOR_PROMPT)
        prompt = prompt_template.format(input=to_be_translate)

        if "verbose" in kwargs.keys():
            if kwargs["verbose"]:
                with get_openai_callback() as cb:
                    _resp = llm.invoke(input=prompt)
                    logger.info(f"Token usage: {cb}")
                    return _resp.content

        _resp = llm.invoke(input=prompt)
        return _resp.content


class YoudaoTranslator(BaseTranslator):
    """A translator that uses Youdao API to translate the input text to English."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translate_source = "youdao"
        self.yd_base_url = "https://openapi.youdao.com/api"
        self.yd_app_id = YD_APP_ID
        self.yd_app_secret_key = YD_APP_SECRET_KEY

    def youdao_get_url_encoded_params(self, to_be_translate):
        """Create Youdao params url encoded

        Args:
            to_be_translate: the string to be translated

        Returns:
            The params which url needs to be encoded
        """
        # the random number to be used to generate the sign
        salt = str(round(time.time() * 1000))
        sign_raw = self.yd_app_id + to_be_translate + salt + self.yd_app_secret_key
        sign = hashlib.md5(sign_raw.encode("utf8")).hexdigest()
        params = {
            "q": to_be_translate,
            "from": "auto",
            "to": "en",
            "appKey": self.yd_app_id,
            "salt": salt,
            "sign": sign,
        }
        return params

    def translate(self, to_be_translate: str, *args, **kwargs) -> str:
        params = self.youdao_get_url_encoded_params(to_be_translate=to_be_translate)
        header = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.get(self.yd_base_url, headers=header, params=params, timeout=7).text
        json_data = json.loads(response)
        trans_text = json_data["translation"][0]
        return trans_text
