import json
import streamlit as st
from typing import Any, Optional
from loguru import logger

from mulcrr.systems.base import System
from mulcrr.agents import Agent, Manager, Analyst, Evaluator, Supervisor, Retriever, Hallucination, Explainer
from mulcrr.utils import parse_answer, parse_action, format_chat_history, parse_json

class CollaborationSystem(System):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['pr']

    def init(self, *args, **kwargs) -> None:
        """
        Initialize the ReAct system.
        """
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        self.init_agents(self.config['agents'])
        self.manager_kwargs = {
            'max_step': self.max_step,
        }
        if self.supervisor is not None:
            self.manager_kwargs['supervisions'] = ''

    def init_agents(self, agents: dict[str, dict]) -> None:
        self.agents: dict[str, Agent] = dict()
        for agent, agent_config in agents.items():
            try:
                agent_class = globals()[agent]
                assert issubclass(agent_class, Agent), f'Agent {agent} is not a subclass of Agent.'
                self.agents[agent] = agent_class(**agent_config, **self.agent_kwargs)
            except KeyError:
                raise ValueError(f'Agent {agent} is not supported.')
        assert 'Manager' in self.agents, 'Manager is required.'
    
    @property
    def manager(self) -> Optional[Manager]:
        if 'Manager' not in self.agents:
            return None
        return self.agents['Manager']
    
    @property
    def analyst(self) -> Optional[Analyst]:
        if 'Analyst' not in self.agents:
            return None
        return self.agents['Analyst']
    
    @property
    def evaluator(self) -> Optional[Evaluator]:
        if 'Evaluator' not in self.agents:
            return None
        return self.agents['Evaluator']

    @property
    def supervisor(self) -> Optional[Supervisor]:
        if 'Supervisor' not in self.agents:
            return None
        return self.agents['Supervisor']
    
    @property
    def retriever(self) -> Optional[Retriever]:
        if 'Retriever' not in self.agents:
            raise None
        return self.agents['Retriever']
    
    @property
    def hallucination(self) -> Optional[Hallucination]:
        if 'Hallucination' not in self.agents:
            raise None
        return self.agents['Hallucination']

    @property
    def explainer(self) -> Optional[Explainer]:
        if 'Explainer' not in self.agents:
            raise None
        return self.agents['Explainer']

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.step_n: int = 1
        if clear:
            if self.supervisor is not None:
                self.supervisor.supervisions = []
                self.supervisor.supervisions_str = ''
            if self.task == 'chat':
                self._chat_history = []
    
    def add_chat_history(self, chat: str, role: str) -> None:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        self._chat_history.append((chat, role))
        
    @property
    def chat_history(self) -> list[tuple[str, str]]:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        return format_chat_history(self._chat_history)
    
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished
        
    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer if self.task != 'chat' else '', json_mode=self.manager.json_mode, mode_input=self.mode_input, **self.kwargs)

    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)

    def act(self) -> tuple[str, Any]:
        # Act
        if self.max_step <= self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        # self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        self.scratchpad += f" {action_type} {str(argument)}"
        logger.debug(f'Action {self.step_n}: {action_type} {str(argument)}')
        return action_type, argument
    
    def execute(self, action_type: str, argument: Any):
        # Execute
        log_head = ''
        omit = False
        hallucination = 'No hallucination'
        if action_type.lower() == 'finish':
            parse_result = self._parse_answer(argument)
            if parse_result['valid']:
                observation = self.finish(parse_result['answer'])
                log_head = 'Finish with answer:\n- '
            else:
                assert "message" in parse_result, "Invalid parse result."
                observation = f'Generated answer is invalid. {parse_result["message"]}\nValid Action examples are as following:\n{self.manager.valid_action_example}.'
        elif action_type.lower() == 'analyse':
            if self.analyst is None:
                observation = 'Analyst is not configured. Cannot execute the action "Analyse".'
            else:
                self.log(f'Calling Analyst with {argument} ...', agent=self.manager, logging=False)
                observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f'Response from Analyst with {argument}:\n- '
                omit = True

                hallucination = self.hallucination_correct("analyse", history=self.analyst.history)
        elif action_type.lower() == 'evaluate':
            if self.evaluator is None:
                observation = 'Evaluator is not configured. Cannot execute the action "Evaluate".'
            else:
                self.log(f'Calling Evaluator with {argument} ...', agent=self.manager, logging=False)
                observation = self.evaluator.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f'Response from Evaluator with {argument}:\n- '
                omit = True

                hallucination = self.hallucination_correct("evaluate", history=self.evaluator.history)
        elif action_type.lower() == 'retrieve':
            if self.retriever is None:
                observation = 'Retriever is not configured. Cannot execute the action "Retrieve".'
            else:
                self.log(f'Calling Retriever with [{argument}] ...', agent=self.manager, logging=False)
                observation = self.retriever.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f'Response from Retriever with [{argument}]:\n- '
                omit = True

                hallucination = self.hallucination_correct("retrieve", response=observation)
        else:
            observation = 'Invalid Action type or format. Valid Action examples are {self.manager.valid_action_example}.'

        self.scratchpad += f'\nObservation: {observation}'

        logger.debug(f'Observation: {observation}')
        self.log(f'{log_head}{observation}', agent=self.manager, logging=False, omit=omit)
        if action_type.lower() != 'finish':
            self.scratchpad += f'\nHallucination: {hallucination}'
            logger.debug(f'Hallucination: {hallucination}')
            if self.hallucination.json_mode:
                self.log(f"{parse_json(hallucination, 'type')}\n- {parse_json(hallucination, 'content')}", agent=self.hallucination, logging=False)
            else:
                self.log(f"{hallucination}", agent=self.hallucination, logging=False)

    def step(self):
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1

    def supervise(self, round_max, round) -> bool:
        correctness = False
        if (not self.is_finished() and not self.is_halted()) or self.supervisor is None:
            self.supervised = False
            if self.supervisor is not None:
                self.manager_kwargs['supervisions'] = ''
            return False
        self.supervisor(self.input, self.scratchpad)
        self.supervised = True
        self.manager_kwargs['supervisions'] = self.supervisor.supervisions_str
        if self.supervisor.json_mode:
            supervision_json = json.loads(self.supervisor.supervisions[-1])
            if 'correctness' in supervision_json and supervision_json['correctness'] == True:
                # don't forward if the last supervision is correct
                logger.debug(f"Last supervision is correct, don't forward.")
                self.log(f"**Last recommendation is correct, don't forward**", agent=self.supervisor, logging=False)
                correctness = True

        if correctness is True or round + 1 >= round_max:
            explanation = self.explainer(self.input, self.scratchpad)
            self.scratchpad += f'\nExplanation: {explanation}'
            logger.debug(f'Explanation: {explanation}')
            self.log(f"Explanation for the answer: " + explanation, agent=self.explainer, logging=False)
        return correctness


    def hallucination_correct(self, prompt: str, **kwargs) -> str:
        if self.hallucination is None:
            hallucination = 'Hallucination is not configured. Cannot execute "Hallucination".'
        else:
            hallucination = self.hallucination(prompt=prompt, **kwargs)

        return hallucination

    def forward(self, round_max: int, round: int, user_input: Optional[str] = None, reset: bool = True, mode_input: bool = False) -> Any:
        self.manager_kwargs['input'] = self.input
        self.mode_input = mode_input
        if reset:
            self.reset()
        while not self.is_finished() and not self.is_halted():
            self.step()
            if self.web_demo and not self.is_finished() and not self.is_halted():
                st.markdown("---")
        self.supervise(round_max, round)
        return self.answer


