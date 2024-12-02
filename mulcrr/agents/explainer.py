from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate

from mulcrr.agents.base import Agent
from mulcrr.utils import format_step, read_json, get_rm


class Explainer(Agent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        self.explainer = self.get_LLM(config=config)
        self.json_mode = self.explainer.json_mode

    @property
    def explainer_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['explainer_prompt_json']
        else:
            return self.prompts['explainer_prompt']

    @property
    def explainer_examples(self) -> str:
        prompt_name = 'explainer_examples_json' if self.json_mode else 'explainer_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''

    def _build_explainer_prompt(self, **kwargs) -> str:
        return self.explainer_prompt.format(
            examples=self.explainer_examples,
            **kwargs
        )

    def _prompt_explainer(self, **kwargs) -> str:
        explainer_prompt = self._build_explainer_prompt(**kwargs)
        explainer_response = self.explainer(explainer_prompt)
        return format_step(explainer_response)

    def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
        logger.trace('Running Reason Explainer...')

        self.explainer_result = self._prompt_explainer(input=input, scratchpad=scratchpad)

        logger.trace(self.explainer_result)
        return self.explainer_result
