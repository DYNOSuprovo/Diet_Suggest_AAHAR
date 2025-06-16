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
    You are an AI assistant specialized in Indian diet and nutrition.
    Based on the following conversation history and the user's query, provide a simple, practical, and culturally relevant **{dietary_type}** food suggestion suitable for Indian users aiming for **{goal}**.
    If a specific region like **{region}** is mentioned or inferred, prioritize food suggestions from that region.
    Focus on readily available ingredients and common Indian dietary patterns for the specified region.
    Be helpful, encouraging, and specific where possible.
    Use the chat history to understand the context of the user's current query and maintain continuity.
    Strictly adhere to the **{dietary_type}** and **{goal}** requirements, and the **{region}** preference if specified.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    User Query:
    {query}

    {dietary_type} {goal} Food Suggestion (Tailored for {region} Indian context):
    """

    return PromptTemplate(
        template=template_string,
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "context"]
    )


# NOTE: In your older code, define_generic_prompt was not explicitly defined here
# I am re-adding it for future compatibility if you wish to use it, but it might not be called directly
# by the old FastAPI routing logic.
def define_generic_prompt():
    """
    Defines a prompt template for handling generic, non-diet-specific user queries.
    """
    template_string = """
    You are an AI assistant specialized in Indian diet and nutrition.
    The user's query is '{query}'. This query is not a direct request for a diet plan but a general question or greeting.

    Respond briefly and politely. You can state your purpose (providing Indian diet suggestions) and offer to help with their dietary goals or preferences. Do not generate a diet plan unless explicitly asked.

    Your response:
    """
    return PromptTemplate(
        template=template_string,
        input_variables=["query"]
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
            | StrOutputParser() # This was the end of your original qa_chain
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
        output_messages_key="answer" # This key was present in your older code too.
    )
    logging.info("Conversational QA Chain initialized.")
    return conversational_qa_chain


def define_merge_prompt_templates():
    # Your older merge prompts had fewer input variables.
    # The default temperature for llm_gemini was 0.5 in your old FastAPI.
    # This might have been too low for the merge if it needed more creativity.
    merge_prompt_default_template = """
    You are an AI assistant specializing in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.

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
    """
    
    merge_prompt_table_template = """
    You are an AI assistant specializing in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.
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
    """

    logging.info("Merge Prompt templates created.")
    return (
        PromptTemplate(template=merge_prompt_default_template, input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"]),
        PromptTemplate(template=merge_prompt_table_template, input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"])
    )
