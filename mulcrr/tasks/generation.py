import os
import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from mulcrr.tasks.base import Task
from mulcrr.utils import init_openai_api, read_json
from mulcrr.systems import CollaborationSystem

class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--dataset', type=str, default='None', help='Dataset name')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='collaboration', choices=['collaboration'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--task', type=str, default='pr', choices=['pr'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        return parser
    
    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = pd.read_csv(data_file)
        if self.task == 'pr':
            candidate_example: str = df['candidate_reviewer_id'][0]
            self.n_candidate = len(candidate_example.split(','))
            self.system_kwargs['n_candidate'] = self.n_candidate # Add n_candidate to system_kwargs by data sample
        return df
    
    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        data_prompt = self.system.prompts[f'data_prompt']
        if self.task == 'pr':
            candidate_example: str = df['candidate_reviewer_id'][0]
            self.n_candidate = len(candidate_example.split(','))
            return [(data_prompt.format(
                PR_id=df['PR_id'][i],
                files=df['files'][i],
                subject=df['subject'][i],
                project=df['project'][i],
                owner_profile=df['owner_profile'][i],
                candidate_reviewer_id=df['candidate_reviewer_id'][i],
            ), df['reviewer_id'][i], df.iloc[i]) for i in tqdm(range(len(df)), desc="Loading data")]
        else:
            raise NotImplementedError
        
    def get_system(self, system: str, system_config: str):
        if system == 'collaboration':
            self.system = CollaborationSystem(config_path=system_config, **self.system_kwargs)
        else:
            raise NotImplementedError
        
    @property
    @abstractmethod
    def running_rounds(self) -> int:
        """Return the rounds to run for each trial.
        
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        Returns:
            `int`: The rounds to run for each trial.
        """
        raise NotImplementedError
    
    @abstractmethod
    def before_generate(self) -> None:
        """The process to run before generating.
        
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError
    
    @abstractmethod
    def after_round(self, answer: Any, gt_answer: int | float | str, round: int, record: dict) -> None:
        """The process to run after each system round during one trial.
        
        Args:
            `answer` (`Any`): The answer given by the system.
            `gt_answer` (`int | float | str`): The ground truth answer.
            `round` (`int`): The current round. Starts from 0.
            `record` (`dict`): The record of the current trial. Can be used to store intermediate results.
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError
    
    @abstractmethod
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        """The process to run after each trial.
        
        Args:
            `answer` (`Any`): The final answer given by the system.
            `gt_answer` (`int | float | str`): The ground truth answer.
            `record` (`dict`): The record of the current trial. Can be used to store intermediate results.
            `pbar` (`tqdm`): The progress bar. Can be used to update the information of the progress bar.
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError
    
    @abstractmethod
    def after_generate(self) -> None:
        """The process to run after generating.
        
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError
    
    def generate(self, data: list[tuple[str, int | float | str, pd.Series]], rounds: int = 2):
        self.before_generate()
        with tqdm(total=len(data)) as pbar:
            for num, (test_data, gt_answer, data_sample) in enumerate(data):
                if num > 40:
                    break
                record = dict()
                self.system.set_data(input=test_data, context="", gt_answer=gt_answer, data_sample=data_sample)
                self.system.reset(clear=True)
                for i in range(rounds):
                    logger.debug(f'===================================Running round {i}...===================================')
                    self.after_round(answer=self.system(rounds, i), gt_answer=gt_answer, round=i, record=record)
                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                pbar.update(1)
        self.after_generate()
    
    def run(self, api_config: str, dataset: str, data_file: str, system: str, system_config: str, task: str, max_his: int):
        if dataset == 'None':
            dataset = os.path.basename(os.path.dirname(data_file))
        self.dataset = dataset
        self.task = task
        self.max_his = max_his
        self.system_kwargs = {
            'task': self.task,
            'leak': False,
            'dataset': self.dataset,
        }
        init_openai_api(read_json(api_config))
        data_df = self.get_data(data_file, max_his)
        self.get_system(system, system_config)
        data = self.prompt_data(data_df)
        self.generate(data, rounds=self.running_rounds)
