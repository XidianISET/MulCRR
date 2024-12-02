from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate

from mulcrr.agents.base import Agent
from mulcrr.utils import format_step, read_json


class Hallucination(Agent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        self.hallucination = self.get_LLM(config=config)
        self.json_mode = self.hallucination.json_mode

    @property
    def hallucination_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['hallucination_prompt_json']
        else:
            return self.prompts['hallucination_prompt']

    @property
    def hallucination_analyse_prompt(self) -> PromptTemplate:
        return self.prompts['hallucination_analyse_prompt_json']

    @property
    def hallucination_evaluate_prompt(self) -> PromptTemplate:
        return self.prompts['hallucination_evaluate_prompt_json']

    @property
    def hallucination_retrieve_prompt(self) -> PromptTemplate:
        return self.prompts['hallucination_retrieve_prompt_json']

    @property
    def hallucination_examples(self) -> str:
        prompt_name = 'hallucination_examples_json' if self.json_mode else 'hallucination_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''

    def _build_hallucination_prompt(self, prompt: str, **kwargs) -> str:
        if prompt == 'analyse':
            return self.hallucination_analyse_prompt.format(
                examples=self.hallucination_examples,
                **kwargs
            )
        if prompt == 'evaluate':
            return self.hallucination_evaluate_prompt.format(
                examples=self.hallucination_examples,
                **kwargs
            )
        if prompt == 'retrieve':
            return self.hallucination_retrieve_prompt.format(
                examples=self.hallucination_examples,
                **kwargs
            )

    def _prompt_hallucination(self, prompt: str, **kwargs) -> str:
        hallucination_prompt = self._build_hallucination_prompt(prompt, **kwargs)
        hallucination_response = self.hallucination(hallucination_prompt)
        return format_step(hallucination_response)

    def forward(self, prompt: str, *args, **kwargs) -> str:
        logger.trace('Running Hallucination...')

        self.hallucination_result = self._prompt_hallucination(prompt, **kwargs)

        logger.trace(self.hallucination_result)
        return self.hallucination_result
