import os
import jsonlines
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from mulcrr.tasks.generation import GenerationTask
from mulcrr.utils import str2list, NumpyEncoder
from mulcrr.evaluation import MetricDict, HitRatioAt, NDCGAt

class EvaluateTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--rounds', type=int, default=2, help='Number of max rounds')
        parser.add_argument('--topks', type=str2list, default=[1, 3, 5], help='Top-Ks for metric')
        return parser
        
    def get_metrics(self, topks: list[int] = [1, 3, 5]):
        if self.task == 'pr':
            self.metrics = MetricDict({
                'true_hit_rate': HitRatioAt(topks=topks),
                'true_ndcg': NDCGAt(topks=topks),
                'valid_hit_rate': HitRatioAt(topks=topks),
                'valid_ndcg': NDCGAt(topks=topks),
            })
        else:
            raise NotImplementedError
    
    def update_evaluation(self, answer: float | int | str, gt_answer: float | int | str) -> str:
        valid = self.system.finished
        logger.debug(f'Answer: {answer}, Ground Truth: {gt_answer}')
        if valid:
            return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            })
        else:
            return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            }, prefix='true')
    
    @property
    def running_rounds(self):
        return self.rounds
    
    def before_generate(self) -> None:
        self.get_metrics(self.topks)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dataset = os.path.basename(os.path.dirname(self.args.data_file))
        data_file = os.path.basename(self.args.data_file)
        run_dir = os.path.join(root_dir, 'run', dataset, self.task, self.args.system)
        os.makedirs(run_dir, exist_ok=True)
        output_args = {
            'data_file': data_file,
            'sampled': self.sampled if hasattr(self, 'sampled') else False,
            'config': self.args.system_config.replace('/', '-'),
            'max_his': self.args.max_his,
            'topk': self.topks,
        }
        output_file_name = '_'.join([f'{k}={v}' for k, v in output_args.items()]) + '.jsonl'
        self.output_file = jsonlines.open(os.path.join(run_dir, output_file_name), mode="w", dumps=NumpyEncoder(ensure_ascii=False).encode, flush=True)
        
    def after_round(self, answer: Any, gt_answer: int | float | str, round: int, record: dict) -> None:
        record[f'Answer_{round}'] = answer
        if hasattr(self.system, 'supervised') and self.system.supervised and self.system.supervisor.keep_supervise:
            logger.trace(f"Supervisor input: {self.system.supervisor.supervisor_input}")
            logger.trace(f"Supervisor output: {self.system.supervisor.supervisor_output}")
    
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        record['Answer_GT'] = gt_answer
        record['type'] = self.system.data_sample['project_parent']
        self.output_file.write(record)
        pbar.set_description(self.update_evaluation(answer, gt_answer))
    
    def after_generate(self) -> None:
        self.output_file.close()
        logger.success("===================================Evaluation Report===================================")
        self.metrics.report()
    
    def run(self, rounds: int, topks: list[int], *args, **kwargs):
        assert kwargs['task'] == 'pr', "Only support reviewer recommendation task."
        self.rounds = rounds
        self.topks = topks
        super().run(*args, **kwargs)

if __name__ == '__main__':
    EvaluateTask().launch()
