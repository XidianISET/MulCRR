import pandas as pd
from typing import Optional

from mulcrr.tools.base import Tool

class InteractionRetriever(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        data_path = self.config['data_path']
        assert data_path is not None, 'Data path not found in config.'
        self.data = pd.read_csv(data_path, sep=',')
        self.data = self.data.sort_values(by=['grant_time'], kind='mergesort')
        assert 'PR_id' in self.data.columns, 'PR_id not found in data.'
        assert 'reviewer_id' in self.data.columns, 'reviewer_id not found in data.'
        self.PR_history = None
        self.reviewer_history = None
    
    def reset(self, submit_time: Optional[int] = None, *args, **kwargs) -> None:
        if submit_time is not None:
            partial_data = self.data[self.data['grant_time'] < submit_time]
            self.PR_history = partial_data.groupby('PR_id')['reviewer_id'].apply(list).to_dict()
            self.PR_files = partial_data.groupby('PR_id')['files'].apply(list).to_dict()
            self.reviewer_history = partial_data.groupby('reviewer_id')['PR_id'].apply(list).to_dict()
            self.reviewer_files = partial_data.groupby('reviewer_id')['files'].apply(list).to_dict()
            self.reviewer_project = partial_data.groupby('reviewer_id')['project'].apply(list).to_dict()
            self.reviewer_subject = partial_data.groupby('reviewer_id')['subject'].apply(list).to_dict()
            self.reviewer_duration = partial_data.groupby('reviewer_id')['duration'].apply(list).to_dict()
            self.reviewer_date = partial_data.groupby('reviewer_id')['grant_date'].apply(list).to_dict()
        else:
            self.partial_data = None
            self.PR_history = None
            self.PR_files = None
            self.reviewer_history = None
            self.reviewer_files = None
            self.reviewer_project = None
            self.reviewer_subject = None
            self.reviewer_duration = None
            self.reviewer_date = None

    def pr_retrieve(self, PR_id: int, k: int, *args, **kwargs) -> str:
        if self.PR_history is None:
            raise ValueError('PR history not found. Please reset the PR_id and reviewer_id.')
        if PR_id not in self.PR_history:
            return f'No history found for PR {PR_id}.'
        retrieved_id = self.PR_history[PR_id][-k:]
        retrieved_files = self.PR_files[PR_id][-k:]
        return f'Retrieved {len(retrieved_id)} reviewers that PR {PR_id} interacted with before: {", ".join(map(str, retrieved_id))} with files: {", ".join(map(str, retrieved_files))}'

    def reviewer_retrieve(self, reviewer_id: int, k: int, *args, **kwargs) -> str:
        if self.reviewer_history is None:
            raise ValueError('Reviewer history not found. Please reset the PR_id and reviewer_id.')
        if reviewer_id not in self.reviewer_history:
            return f'No history found for reviewer {reviewer_id}.'
        retrieved_id = self.reviewer_history[reviewer_id][-k:]
        retrieved_files = self.reviewer_files[reviewer_id][-k:]
        retrieved_project = self.reviewer_project[reviewer_id][-k:]
        retrieved_subject = self.reviewer_subject[reviewer_id][-k:]
        retrieved_duration = self.reviewer_duration[reviewer_id][-k:]
        retrieved_date = self.reviewer_date[reviewer_id][-k:]
        return f'Retrieved {len(retrieved_id)} PRs that interacted with reviewer {reviewer_id} before: {", ".join(map(str, retrieved_id))}. **Projects**: {", ".join(map(str, retrieved_project))}. **Subjects**: {", ".join(map(str, retrieved_subject))}. **Files**: {", ".join(map(str, retrieved_files))}. **Duration**: {", ".join(map(str, retrieved_duration))} seconds. **Review date**: {", ".join(map(str, retrieved_date))}'
