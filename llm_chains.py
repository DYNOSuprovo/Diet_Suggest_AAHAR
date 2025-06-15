# llm_chains.py

import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    template_string = """
    You are AAHAR, an AI assistant specialized in Indian diet and nutrition, created by Suprovo Mallick.

    You have two responsibilities:
    1. If the user query is related to food, diet, health goals, or nutrition, provide a culturally relevant Indian food suggestion tailored for **{dietary_type}** users aiming for **{goal}**. Use chat history and regional context (**{region}**) to improve your answer.
    2. If the user query is a general conversation (like greetings, who created you, your name, or what you do), politely respond with relevant information and DO NOT return a food suggestion.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    User Query:
    {query}

    Output:
    """
    return PromptTemplate(
        template=template_string,
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "context"]
    )



def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    try:
        retriever = db.as_retriever(search_kwargs={"k": 5})

        def retrieve_and_log_context(input_dict):
            docs = retriever.invoke(input_dict["query"])
            if not docs:
                logging.warning(f"No documents retrieved for query: '{input_dict['query']}'")
            context_str = "\n\n".join(doc.page_content for doc in docs)
            logging.info(f"Retrieved Context: {context_str}")
            return context_str

        qa_chain = (
            {
                "context": retrieve_and_log_context,
                "query": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "dietary_type": RunnablePassthrough(),
                "goal": RunnablePassthrough(),
                "region": RunnablePassthrough(),
            }
            | rag_prompt
            | llm_gemini
            | StrOutputParser()
        )
        logging.info("Retrieval QA Chain initialized successfully.")
        return qa_chain
    except Exception as e:
        logging.exception("Full QA Chain setup traceback:")
        raise RuntimeError(f"QA Chain setup error: {e}")


def setup_conversational_qa_chain(qa_chain):
    conversational_qa_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    logging.info("Conversational QA Chain initialized.")
    return conversational_qa_chain


def define_merge_prompt_templates():
    merge_prompt_default = PromptTemplate.from_template("""
    You are an AI assistant specializing in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical **{dietary_type}** food suggestion or diet plan for **{goal}**, tailored for a **{region}** Indian context.

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
    3. Combine information logically and seamlessly, without mentioning the source of each piece.
    4. Ensure the final plan is clear, actionable, and culturally relevant.
    5. If the user's input was only a greeting, respond politely without providing a diet plan.

    Final {dietary_type} {goal} Food Suggestion/Diet Plan (Tailored for {region} Indian context):
    """)

    merge_prompt_table = PromptTemplate.from_template("""
    You are an AI assistant specializing in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical **{dietary_type}** food suggestion or diet plan for **{goal}**, tailored for a **{region}** Indian context.
    **You MUST present the final diet plan as a clear markdown table. Include columns for Meal, Food Items, and Notes/Considerations.**

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
    3. Combine information logically and seamlessly, without mentioning the source of each piece.
    4. Ensure the final plan is clear, actionable, and culturally relevant.
    5. If the user's input was only a greeting, respond politely without providing a diet plan.

    Final {dietary_type} {goal} Diet Plan (Tailored for {region} Indian context, in markdown table format):
    """)

    logging.info("Merge Prompt templates created.")
    return merge_prompt_default, merge_prompt_table
