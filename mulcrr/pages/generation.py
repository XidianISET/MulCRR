import os
import pandas as pd
import streamlit as st

from mulcrr.systems import System
from mulcrr.utils import add_chat_message

@st.cache_data
def read_data(file_path: str):
    return pd.read_csv(file_path)


def on_click():
    if st.session_state.index_current != -1 and st.session_state.index_current != st.session_state.index:
        st.session_state.round = -1
    st.session_state.index_current = st.session_state.index
    st.session_state.round += 1


def gen_page(system: System, task: str, dataset: str):
    data = read_data(os.path.join('data', dataset, f'test.csv'))
    length = len(data)

    if 'index' not in st.session_state:
        st.session_state.index = -1
    if 'index_current' not in st.session_state:
        st.session_state.index_current = None
    if 'round' not in st.session_state:
        st.session_state.round = -1
    st.session_state.index = st.number_input('Choose a sample', 1, length, 1)
    reset_data = False
    if 'data_sample' not in st.session_state:
        st.session_state.data_sample = f'{dataset}_{st.session_state.index}'
        reset_data = True
    elif st.session_state.data_sample != f'{dataset}_{st.session_state.index}':
        st.session_state.data_sample = f'{dataset}_{st.session_state.index}'
        reset_data = True
    data_sample = data.iloc[st.session_state.index - 1]
    data_prompt = system.prompts[f'data_prompt']
    with st.expander('Data Sample', expanded=True):
        st.markdown(f'#### Data Sample: {st.session_state.index} / {length}')
        st.markdown(f'##### PR ID: {data_sample["PR_id"]}')
        st.markdown(f'##### PR Info:')
        st.markdown(f'```\n{data_sample["PR_info"]}\n```')
        if task == 'pr':
            st.markdown(f'##### Candidate Reviewer History:')
            data_sample_candidates = data_sample['candidate_reviewer_id'].split(',')
            system.kwargs['n_candidate'] = len(data_sample_candidates)
            data_sample_candidates = [f'{i + 1}. {line}' for i, line in enumerate(data_sample_candidates)]
            data_sample_candidates = '\n'.join(data_sample_candidates)
            st.markdown(f'```\n{data_sample_candidates}\n```')
            system_input = data_prompt.format(
                PR_id=data_sample['PR_id'],
                files=data_sample['files'],
                project=data_sample['project'],
                subject=data_sample['subject'],
                owner_profile=data_sample['owner_profile'],
                candidate_reviewer_id=data_sample['candidate_reviewer_id'],
            )
            gt_answer = data_sample['reviewer_id']
            st.markdown(f'##### Reviewer ID (Ground Truth): {data_sample["reviewer_id"]}')

        else:
            raise NotImplementedError
    if reset_data:
        system.set_data(input=system_input, context='', gt_answer=gt_answer, data_sample=data_sample)
        system.reset(clear=True)
        st.session_state.chat_history = []
        st.session_state.round = -1
    for chat in st.session_state.chat_history:
        if isinstance(chat['message'], str):
            st.chat_message(chat['role']).markdown(chat['message'])
        elif isinstance(chat['message'], list):
            with st.chat_message(chat['role']):
                for message in chat['message']:
                    st.markdown(f'{message}')
        else:
            raise ValueError
    max_round = st.number_input('Max round', 1, 5, 1)
    st.markdown(f"#### Current Sample: {st.session_state.index_current}")
    if st.button('Start one round', on_click=on_click, disabled=(max_round <= st.session_state.round + 1)):
        with st.chat_message('assistant'):
            title = f'#### System running round {st.session_state.round + 1}'
            st.markdown(title)
            answer = system(max_round, st.session_state.round)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'message': [title] + system.web_log
            })
        if task == 'pr':
            add_chat_message('assistant', f'**Answer**: `{answer}`, Ground Truth: `{gt_answer}`')
        st.session_state.start_round = False
        st.rerun()
