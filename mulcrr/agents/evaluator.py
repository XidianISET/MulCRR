from typing import Any
from loguru import logger

from mulcrr.agents.base import ToolAgent
from mulcrr.tools import InfoDatabase, InteractionRetriever
from mulcrr.utils import read_json, get_rm

class Evaluator(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.evaluator = self.get_LLM(config=config)
        self.json_mode = self.evaluator.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
        }

    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']

    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']

    @property
    def evaluator_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['evaluator_prompt_json']
        else:
            return self.prompts['evaluator_prompt']

    @property
    def evaluator_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['evaluator_fewshot_json']
        else:
            return self.prompts['evaluator_fewshot']

    def _build_evaluator_prompt(self, **kwargs) -> str:
        return self.evaluator_prompt.format(
            fewshot=self.evaluator_fewshot,
            history=self.history,
            **kwargs
        )

    def _prompt_evaluator(self, **kwargs) -> str:
        evaluator_prompt = self._build_evaluator_prompt(**kwargs)
        evaluation = self.evaluator(evaluator_prompt)
        return evaluation

    def command(self, action_type: str, id: int, k: int = 5) -> None:
        log_head = ''
        head = "ERROR"
        if action_type.lower() == 'info':
            try:
                logger.debug(f'Action: Reviewer info')
                observation = self.info_retriever.reviewer_info(reviewer_id=id)
                log_head = f'Look up ReviewerInfo of reviewer {id} ...\n- '
                head = "Info"
            except ValueError or TypeError:
                observation = f"Invalid reviewer id: {id}"
        elif action_type.lower() == 'history':
            if isinstance(id, int) and isinstance(k, int):
                logger.debug(f'Action: Reviewer interaction history')
                observation = self.interaction_retriever.reviewer_retrieve(reviewer_id=id, k=k)
                log_head = f'Look up ReviewerHistory of reviewer {id} with at most {k} PRs ...\n- '
                head = "History"
            else:
                observation = f"Invalid reviewer id and retrieval number: {id}"
        elif action_type.lower() == 'finish':
            logger.debug(f'Finish Evaluation')
            results = self._prompt_evaluator(id=id)
            observation = self.finish(results=results)
            log_head = 'Finish with results:\n- '
            head = "Evaluation"
        else:
            observation = f'Unknown command type: {action_type}.'
        logger.debug(f'{head}: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': action_type,
            'head': head,
            'observation': observation,
        }
        self._history.append(turn)

    def forward(self, id: int, *args: Any, **kwargs: Any) -> str:
        assert self.system.data_sample is not None, "Data sample is not provided."
        assert 'submit_time' in self.system.data_sample, "Submit date is not provided."
        self.interaction_retriever.reset(submit_time=self.system.data_sample['submit_time'])

        self.command('info', id)
        self.command('history', id, 5)
        self.command('finish', id)
        if not self.finished:
            return "Evaluator did not return any result."
        return self.results

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if json_mode:
            if not isinstance(argument, list) or len(argument) != 2:
                observation = "The argument of the action 'Evaluate' should be a list with two elements: type (reviewer) and id."
                return observation
            else:
                type, id = argument
                if type.lower() != 'reviewer':
                    observation = f"Invalid type: {type}. It should be 'reviewer'."
                    return observation
                elif not isinstance(id, int):
                    observation = f"Invalid id: {id}. It should be an integer."
                    return observation
        else:
            if len(argument.split(',')) != 2:
                observation = "The argument of the action 'Evaluate' should be a string with two elements separated by a comma: type (reviewer) and id."
                return observation
            else:
                type, id = argument.split(',')
                if type.lower() != 'reviewer':
                    observation = f"Invalid type: {type}. It should be 'reviewer'."
                    return observation
                else:
                    try:
                        id = int(id)
                    except ValueError or TypeError:
                        observation = f"Invalid id: {id}. The id should be an integer."
                        return observation
        return self(id=id)
