from typing import Any
from loguru import logger

from mulcrr.agents.base import ToolAgent
from mulcrr.tools import InfoDatabase, InteractionRetriever
from mulcrr.utils import read_json, get_rm

class Analyst(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.analyst = self.get_LLM(config=config)
        self.json_mode = self.analyst.json_mode
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
    def analyst_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_prompt_json']
        else:
            return self.prompts['analyst_prompt']

    @property
    def analyst_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_fewshot_json']
        else:
            return self.prompts['analyst_fewshot']

    def _build_analyst_prompt(self, **kwargs) -> str:
        return self.analyst_prompt.format(
            fewshot=self.analyst_fewshot,
            history=self.history,
            **kwargs
        )

    def _prompt_analyst(self, **kwargs) -> str:
        analyst_prompt = self._build_analyst_prompt(**kwargs)
        analysis = self.analyst(analyst_prompt)
        return analysis

    def command(self, action_type: str, id: int, k: int = 5) -> None:
        log_head = ''
        head = "ERROR"
        if action_type.lower() == 'info':
            try:
                logger.debug(f'Action: Pull Request info')
                observation = self.info_retriever.pr_info(PR_id=id)
                log_head = f'Look up PRInfo of PR {id} ...\n- '
                head = "Info"
            except ValueError or TypeError:
                observation = f"Invalid PR id: {id}"
        elif action_type.lower() == 'history':
            if isinstance(id, int) and isinstance(k, int):
                logger.debug(f'Action: Pull Request interaction history')
                observation = self.interaction_retriever.pr_retrieve(PR_id=id, k=k)
                log_head = f'Look up PRHistory of PR {id} with at most {k} reviewers ...\n- '
                head = "History"
            else:
                observation = f"Invalid PR id and retrieval number: {id}"
        elif action_type.lower() == 'finish':
            logger.debug(f'Action: Finish Analysis')
            results = self._prompt_analyst(id=id)
            observation = self.finish(results=results)
            log_head = 'Finish with results:\n- '
            head = "Analysis"
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
        self.command('finish', id)
        if not self.finished:
            return "Analyst did not return any result."
        return self.results

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if json_mode:
            if not isinstance(argument, list) or len(argument) != 2:
                observation = "The argument of the action 'Analyse' should be a list with two elements: type (pullrequest) and id."
                return observation
            else:
                type, id = argument
                if type.lower() != 'pullrequest':
                    observation = f"Invalid type: {type}. It should be 'pullrequest'."
                    return observation
                elif not isinstance(id, int):
                    observation = f"Invalid id: {id}. It should be an integer."
                    return observation
        else:
            if len(argument.split(',')) != 2:
                observation = "The argument of the action 'Analyse' should be a string with two elements separated by a comma: type (pullrequest) and id."
                return observation
            else:
                type, id = argument.split(',')
                if type.lower() != 'pullrequest':
                    observation = f"Invalid type: {type}. It should be 'pullrequest'."
                    return observation
                else:
                    try:
                        id = int(id)
                    except ValueError or TypeError:
                        observation = f"Invalid id: {id}. The id should be an integer."
                        return observation
        return self(id=id)
