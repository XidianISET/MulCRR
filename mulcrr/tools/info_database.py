import pandas as pd

from mulcrr.tools.base import Tool

class InfoDatabase(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pr_info_path = self.config.get('pr_info', None)
        reviewer_info_path = self.config.get('reviewer_info', None)
        if pr_info_path is not None:
            self._pr_info = pd.read_csv(pr_info_path, sep=',')
            assert 'PR_id' in self._pr_info.columns, 'PR_id column not found in PR_info.'
        if reviewer_info_path is not None:
            self._reviewer_info = pd.read_csv(reviewer_info_path, sep=',')
            assert 'reviewer_id' in self._reviewer_info.columns, 'reviewer_id column not found in reviewer_info.'
        
    def reset(self, *args, **kwargs) -> None:
        pass
    
    def pr_info(self, PR_id: int) -> str:
        if not hasattr(self, '_pr_info'):
            return 'PR info database not available.'
        info = self._pr_info[self._pr_info['PR_id'] == PR_id]
        if info.empty:
            return f'PR {PR_id} not found in PR info database.'
        assert len(info) == 1, f'Multiple entries found for PR {PR_id}.'
        if 'PR_info' in self._pr_info.columns:
            return info['PR_info'].values[0].replace('\n', '; ')
        else:
            columns = self._pr_info.columns
            columns = columns.drop('PR_id')
            profile = '; '.join([f'{column}: {info[column].values[0]}' for column in columns])
            return f'PR {PR_id} Info:\n{profile}'
    
    def reviewer_info(self, reviewer_id: int) -> str:
        if not hasattr(self, '_reviewer_info'):
            return 'Reviewer info database not available.'
        info = self._reviewer_info[self._reviewer_info['reviewer_id'] == reviewer_id]
        if info.empty:
            return f'Reviewer {reviewer_id} not found in reviewer info database.'
        assert len(info) == 1, f'Multiple entries found for reviewer {reviewer_id}.'
        if 'reviewer_profile' in self._reviewer_info.columns:
            return info['reviewer_profile'].values[0].replace('\n', '; ')
        else:
            columns = self._reviewer_info.columns
            columns = columns.drop('reviewer_id')
            profile = '; '.join([f'{column}: {info[column].values[0]}' for column in columns])
            return f'Reviewer {reviewer_id} Profile:\n{profile}'
