import json
import re
from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate

from mulcrr.agents.base import Agent
from mulcrr.utils import read_json, parse_action, get_rm

class Retriever(Agent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        self.retriever = self.get_LLM(config=config)
        self.json_mode = self.retriever.json_mode

    @property
    def retriever_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['retriever_prompt_json']
        else:
            return self.prompts['retriever_prompt']

    @property
    def retriever_examples(self) -> str:
        if self.json_mode:
            return self.prompts['retriever_examples_json']
        else:
            return self.prompts['retriever_examples']

    def parse(self, response: str, json_mode: bool = False) -> str:
        if json_mode:
            try:
                json_response = json.loads(response)
                return f'{json_response["requirement"]}[{json_response["description"]}]\n({json_response["url"]})'
            except:
                return 'Invalid response'
        else:
            return response

    def _build_retriever_prompt(self, **kwargs) -> str:
        return self.retriever_prompt.format(
            examples=self.retriever_examples,
            **kwargs
        )

    def _prompt_retriever(self, **kwargs) -> str:
        retriever_prompt = self._build_retriever_prompt(**kwargs)
        response = self.retriever(retriever_prompt)
        return response

    def forward(self, requirement: str, *args, **kwargs) -> str:
        response = self._prompt_retriever(requirement=requirement)
        response = self.parse(response, self.json_mode)

        self.observation(response, f"Retrieving [{requirement}] ...\n- ")

        return response

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if not isinstance(argument, str):
            return f'Invalid argument type: {type(argument)}. Must be a string.'
        return self(requirement=argument)
