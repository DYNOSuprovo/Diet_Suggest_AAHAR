# streamlit_ui.py
import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = "session_" + os.urandom(8).hex()
        st.session_state.messages = []
        st.session_state.dietary_type = "any"
        st.session_state.diet_goal = "diet"
        st.session_state.region = "Indian"
        st.session_state.table_format_requested = False
        st.session_state.last_substantive_query = ""
        logging.info(f"New Streamlit session: {st.session_state.session_id}")
    else:
        # Ensure all necessary keys are present if session existed before full init
        if 'dietary_type' not in st.session_state: st.session_state.dietary_type = "any"
        if 'diet_goal' not in st.session_state: st.session_state.diet_goal = "diet"
        if 'region' not in st.session_state: st.session_state.region = "Indian"
        if 'table_format_requested' not in st.session_state: st.session_state.table_format_requested = False
        if 'last_substantive_query' not in st.session_state: st.session_state.last_substantive_query = ""
        logging.info(f"Existing session: {st.session_state.session_id}. Type: {st.session_state.dietary_type}, Goal: {st.session_state.diet_goal}, Region: {st.session_state.region}, Table: {st.session_state.table_format_requested}, LastQuery: {st.session_state.last_substantive_query}")


def setup_sidebar():
    """Sets up the Streamlit sidebar elements."""
    with st.sidebar:
        st.markdown("👤 **Created by Lord d'Artagnan**")
        st.markdown("---")
        use_llms_toggle = st.toggle("🔄 Include expanded suggestions", value=True, help="Fetch from LLaMA, Mixtral, Gemma via Groq.")
    return use_llms_toggle

def display_chat_messages():
    """Displays all messages stored in st.session_state.messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def update_ui_for_session_switch(session_id_input: str, get_session_history_func, query_analysis_module):
    """
    Handles session switching: updates session ID, loads history, and infers preferences.
    Requires query_analysis_module to extract preferences from old messages.
    """
    old_session_id = st.session_state.session_id
    st.session_state.session_id = session_id_input
    logging.info(f"User changed session ID from {old_session_id} to: {st.session_state.session_id}")
    current_langchain_history = get_session_history_func(st.session_state.session_id)
    new_ui_messages = []
    # Reset session specifics for the new/switched session
    st.session_state.dietary_type = "any"
    st.session_state.diet_goal = "diet"
    st.session_state.region = "Indian"
    st.session_state.table_format_requested = False
    st.session_state.last_substantive_query = ""

    temp_last_substantive_query = ""
    for message_obj in current_langchain_history.messages:
        if isinstance(message_obj, HumanMessage):
            new_ui_messages.append({"role": "user", "content": message_obj.content})
            # Infer states from historical messages
            hist_diet_type = query_analysis_module.extract_diet_preference(message_obj.content)
            if hist_diet_type != "any": st.session_state.dietary_type = hist_diet_type
            hist_diet_goal = query_analysis_module.extract_diet_goal(message_obj.content)
            if hist_diet_goal != "diet": st.session_state.diet_goal = hist_diet_goal
            hist_region = query_analysis_module.extract_regional_preference(message_obj.content)
            if hist_region != "Indian": st.session_state.region = hist_region
            if query_analysis_module.contains_table_request(message_obj.content): st.session_state.table_format_requested = True

            if not query_analysis_module.is_formatting_request(message_obj.content) and not query_analysis_module.is_greeting(message_obj.content):
                temp_last_substantive_query = message_obj.content
        elif isinstance(message_obj, AIMessage):
            new_ui_messages.append({"role": "assistant", "content": message_obj.content})
    st.session_state.last_substantive_query = temp_last_substantive_query
    st.session_state.messages = new_ui_messages
    logging.info(f"Switched to session {st.session_state.session_id}. Loaded {len(new_ui_messages)} UI messages. Last substantive: '{temp_last_substantive_query}'")
    st.toast(f"Switched to session: {st.session_state.session_id}. History loaded.")
    st.rerun()