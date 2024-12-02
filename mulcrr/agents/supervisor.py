import json

import tiktoken
from enum import Enum
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from mulcrr.agents.base import Agent
from mulcrr.llms import AnyOpenAILLM
from mulcrr.utils import format_step, format_supervisions, format_last_attempt, read_json, get_rm, parse_json

class Strategy(Enum):
    """
    Supervision strategies for the `Supervisor` agent. `NONE` means no supervision. `SUPERVISE` is the default strategy for the `Supervisor` agent, which prompts the LLM to supervise on the input and the scratchpad. `LAST_ATTEMPT` simply store the input and the scratchpad of the last attempt. `LAST_ATTEMPT_AND_SUPERVISE` combines the two strategies.
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    SUPERVISE = 'supervise'
    LAST_ATTEMPT_AND_SUPERVISE = 'last_trial_and_supervise'

class Supervisor(Agent):
    """
    The supervisor agent. The supervisor agent prompts the LLM to supervise on the input and the scratchpad as default. Other supervision strategies are also supported. See `Strategy` for more details.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """Initialize the supervisor agent. The supervisor agent prompts the LLM to supervise on the input and the scratchpad as default.
        
        Args:
            `config_path` (`str`): The path to the config file of the supervisor LLM.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        keep_supervise = get_rm(config, 'keep_supervise', True)
        supervision_strategy = get_rm(config, 'strategy', Strategy.SUPERVISE.value)
        self.llm = self.get_LLM(config=config)
        if isinstance(self.llm, AnyOpenAILLM):
            self.enc = tiktoken.encoding_for_model(self.llm.model_name)
        else:
            self.enc = AutoTokenizer.from_pretrained(self.llm.model_name)
        self.json_mode = self.llm.json_mode
        self.keep_supervise = keep_supervise
        for strategy in Strategy:
            if strategy.value == supervision_strategy:
                self.supervision_strategy = strategy
                break
        assert self.supervision_strategy is not None, f'Unknown supervision strategy: {supervision_strategy}'
        self.supervisions: list[str] = []
        self.supervisions_str: str = ''

    @property
    def supervisor_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['supervisor_prompt_json']
        else:
            return ''
        
    @property
    def supervisor_examples(self) -> str:
        prompt_name = 'supervisor_examples_json' if self.json_mode else 'supervisor_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''

    def parse(self, response: str, json_mode: bool = False) -> str:
        if json_mode:
            try:
                json_response = json.loads(response)
                if 'new_plan' in json_response:
                    return f"{json_response['correctness']}\n- **Reason**: {json_response['reason']}\n- **New Plan**: {json_response['new_plan']}"
                else:
                    return f"{json_response['correctness']}\n- **Reason**: {json_response['reason']}"
            except:
                return 'Invalid response'
        else:
            return response

    def _build_supervisor_prompt(self, input: str, scratchpad: str) -> str:
        return self.supervisor_prompt.format(
            examples=self.supervisor_examples,
            input=input,
            scratchpad=scratchpad
        )
    
    def _prompt_supervision(self, input: str, scratchpad: str) -> str:
        supervisor_prompt = self._build_supervisor_prompt(input, scratchpad)
        supervisor_response = self.llm(supervisor_prompt)
        supervisor_response_json = format_step(supervisor_response)
        supervisor_response_json = self.parse(supervisor_response_json, self.json_mode)
        if self.keep_supervise:
            self.supervisor_input = supervisor_prompt
            self.supervisor_output = supervisor_response_json
            logger.trace(f'Supervisor input length: {len(self.enc.encode(self.supervisor_input))}')
            logger.trace(f"Supervisor input: {self.supervisor_input}")
            logger.trace(f'Supervisor output length: {len(self.enc.encode(self.supervisor_output))}')
            if self.json_mode:
                self.system.log(f"[Correctness]: {self.supervisor_output}", agent=self, logging=False)
            else:
                self.system.log(f"[Correctness]:\n- {self.supervisor_output}", agent=self, logging=False)
            logger.debug(f"Supervisor output: {self.supervisor_output}")
        return format_step(supervisor_response)

    def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
        logger.trace('Running Supervise strategy...')

        if self.supervision_strategy == Strategy.LAST_ATTEMPT:
            self.supervisions = [scratchpad]
            self.supervisions_str = format_last_attempt(input, scratchpad, self.prompts['last_trial_header'])

        elif self.supervision_strategy == Strategy.SUPERVISE:
            self.supervisions.append(self._prompt_supervision(input=input, scratchpad=scratchpad))
            self.supervisions_str = format_supervisions(self.supervisions, header=self.prompts['supervise_header'])

        elif self.supervision_strategy == Strategy.LAST_ATTEMPT_AND_SUPERVISE:
            self.supervisions_str = format_last_attempt(input, scratchpad, self.prompts['last_trial_header'])
            self.supervisions = self._prompt_supervision(input=input, scratchpad=scratchpad)
            self.supervisions_str += format_supervisions(self.supervisions, header=self.prompts['supervise_last_trial_header'])

        elif self.supervision_strategy == Strategy.NONE:
            self.supervisions = []
            self.supervisions_str = ''

        else:
            raise ValueError(f'Unknown supervise strategy: {self.supervision_strategy}')

        logger.trace(self.supervisions_str)
        return self.supervisions_str
