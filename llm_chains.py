# llm_chains.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        logging.info(f"Creating new Langchain session history in 'store' for: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history from 'store' for: {session_id}")
    return store[session_id]

def define_rag_prompt_template():
    prompt = PromptTemplate.from_template("""
    You are an AI assistant specializing in Indian diet and nutrition.
    Given the user's dietary preference (**{dietary_type}**), goal (**{goal}**), and region (**{region}**),
    generate a culturally relevant food suggestion or diet plan.
    Use the chat history and retrieved knowledge base to give a helpful response.

    Chat History:
    {chat_history}

    Context:
    {context}

    Query:
    {query}

    Respond with a clear and concise food recommendation:
    """)
    logging.info("RAG Prompt template created (using 'query').")
    return prompt

def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    chain_kwargs = {
        "prompt": rag_prompt
    }
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gemini,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs=chain_kwargs,
        return_source_documents=False,
        input_key="query"
    )
    logging.info("Retrieval QA Chain initialized successfully (input_key='query').")
    return qa_chain

def setup_conversational_qa_chain(qa_chain: RetrievalQA):
    conversational_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    logging.info("Conversational QA Chain initialized (input_messages_key='query').")
    return conversational_chain

def define_merge_prompt_templates():
    default = PromptTemplate.from_template("""
    Merge the following AI-generated responses into one final, concise and culturally relevant food suggestion.
    Tailor it for a **{dietary_type}** Indian user aiming for **{goal}**, preferably from **{region}**.

    RAG:
    {rag}

    LLaMA:
    {llama}

    Mixtral:
    {mixtral}

    Gemma:
    {gemma}

    Final merged suggestion:
    """)
    
    table = PromptTemplate.from_template("""
    Create a diet table with meal-wise breakdown based on the following suggestions.
    Ensure it's for a **{dietary_type}** user with goal = **{goal}**, region = **{region}**.

    RAG:
    {rag}

    LLaMA:
    {llama}

    Mixtral:
    {mixtral}

    Gemma:
    {gemma}

    Output format: markdown table with Meal | Items | Notes
    """)
    
    logging.info("Merge Prompt templates created.")
    return default, table
