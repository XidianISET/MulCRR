# Description: Answer checking utilities, including normalization and comparison.

import re
import string
from typing import Any

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer: str, key: str) -> bool:
    """Exact match.
    
    Args:
        `answer` (`str`): The answer to be checked.
        `key` (`str`): The ground truth answer.
    Returns:
        `bool`: Whether the answer is correct.
    """
    return normalize_answer(answer) == normalize_answer(key)

def is_correct_pr(answer: list[int], gt_answer: int) -> bool:
    if len(answer) == 0:
        return False
    return answer[0] == gt_answer

def is_correct(task: str, answer: Any, gt_answer: Any) -> bool:
    """Check whether the answer is correct.
    
    Args:
        `task` (`str`): Task type.
        `answer` (`Any`): The answer to be checked.
        `gt_answer` (`Any`): The ground truth answer.
    Raises:
        `ValueError`: Unsupported task type.
    Returns:
        `bool`: Whether the answer is correct.
    """
    if task == 'pr':
        return is_correct_pr(answer, gt_answer)
    else:
        raise ValueError(f'Unsupported task type: {task}')
