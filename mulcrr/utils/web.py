import streamlit as st
from typing import Optional

def add_chat_message(role: str, message: str, avatar: Optional[str] = None):
    """Add a chat message to the chat history.
    
    Args:
        `role` (`str`): The role of the message.
        `message` (`str`): The message to be added.
        `avatar` (`Optional[str]`): The avatar of the agent. If `avatar` is `None`, use the default avatar.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': role, 'message': message})
    if avatar is not None:
        st.chat_message(role, avatar=avatar).markdown(message)
    else:
        st.chat_message(role).markdown(message)

def get_color(agent_type: str) -> str:
    """Get the color of the agent.
    
    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The color name of the agent.
    """
    if 'manager' in agent_type.lower():
        return 'grey'
    elif 'supervisor' in agent_type.lower():
        return 'grey'
    elif 'retriever' in agent_type.lower():
        return 'grey'
    elif 'evaluator' in agent_type.lower():
        return 'grey'
    elif 'analyst' in agent_type.lower():
        return 'grey'
    elif 'explainer' in agent_type.lower():
        return 'grey'
    elif 'hallucination' in agent_type.lower():
        return 'grey'
    else:
        return 'gray'

def get_role(agent_type: str) -> str:
    if 'manager' in agent_type.lower():
        return 'Coordinator'
    elif 'supervisor' in agent_type.lower():
        return 'Supervisor'
    elif 'retriever' in agent_type.lower():
        return 'DataRetriever'
    elif 'analyst' in agent_type.lower():
        return 'CodeAnalyst'
    elif 'evaluator' in agent_type.lower():
        return 'ReviewerEvaluator'
    elif 'explainer' in agent_type.lower():
        return 'ReasonExplainer'
    elif 'hallucination' in agent_type.lower():
        return 'HallucinationDetector'
    else:
        return 'Agent'
