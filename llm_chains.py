import logging
from langchain.chains import RetrievalQA # Keep this for type hinting if needed for clarity elsewhere, but not used directly for chain building
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # <-- NEW IMPORT

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

    diet_prompt = PromptTemplate(
        template=template_string,
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "context"]
    )
    logging.info("RAG Prompt template created (with explicit input_variables).")
    return diet_prompt

# --- MODIFIED AGAIN: setup_qa_chain to properly log context and pass it ---
def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    try:
        retriever = db.as_retriever(search_kwargs={"k": 5})

        # Define a function to retrieve and log context
        def retrieve_and_log_context(input_dict):
            docs = retriever.invoke(input_dict["query"])
            context_str = "\n\n".join(doc.page_content for doc in docs)
            logging.info(f"Retrieved Context: {context_str}")
            return context_str

        # Define the RAG chain using LCEL
        # We need to explicitly pass all inputs to the prompt after context is generated
        qa_chain = (
            {
                "context": retrieve_and_log_context, # Call our custom function to get and log context
                "query": RunnablePassthrough(), # Pass the original query through
                "chat_history": RunnablePassthrough(), # Pass chat_history through
                "dietary_type": RunnablePassthrough(), # Pass dietary_type through
                "goal": RunnablePassthrough(), # Pass goal through
                "region": RunnablePassthrough(), # Pass region through
            }
            | rag_prompt
            | llm_gemini
            | StrOutputParser()
        )
        logging.info("Retrieval QA Chain initialized successfully using LCEL.")
        return qa_chain
    except Exception as e:
        logging.exception("Full QA Chain setup traceback:")
        raise RuntimeError(f"QA Chain setup error: {e}")

# --- setup_conversational_qa_chain and define_merge_prompt_templates remain unchanged ---
def setup_conversational_qa_chain(qa_chain):
    conversational_qa_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer" # This is important as `RunnableWithMessageHistory` expects this key
    )
    logging.info("Conversational QA Chain initialized (input_messages_key='query').")
    return conversational_qa_chain

def define_merge_prompt_templates():
    merge_prompt_template_default = """
    You are a diet planning assistant.
    Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical **{dietary_type}** diet plan or suggestion for **{goal}**, tailored for a **{region}** Indian context if a region is specified.
    Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
    Ensure the final plan is clear, actionable, and tailored for Indian users. Present as a clear list or paragraph.
    If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.

    Primary RAG Answer:
    {rag}

    Additional Suggestions:
    - LLaMA Suggestion: {llama}
    - Mixtral Suggestion: {mixtral}
    - Gemma Suggestion: {gemma}

    Refined and Merged {dietary_type} {goal} Food Suggestion (Tailored for {region} Indian context):
    """
    merge_prompt_default = PromptTemplate.from_template(merge_prompt_template_default)

    merge_prompt_template_table = """
    You are a diet planning assistant.
    Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical **{dietary_type}** diet plan or suggestion for **{goal}**, tailored for a **{region}** Indian context if a region is specified.
    Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
    Ensure the final plan is clear, actionable, and tailored for Indian users.
    **You MUST present the final diet plan as a clear markdown table. Include columns for Meal, Food Items, and Notes/Considerations.**
    If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.

    Primary RAG Answer:
    {rag}

    Additional Suggestions:
    - LLaMA Suggestion: {llama}
    - Mixtral Suggestion: {mixtral}
    - Gemma Suggestion: {gemma}

    Refined and Merged {dietary_type} {goal} Diet Plan/Suggestion (Tailored for {region} Indian context, in markdown table format):
    """
    merge_prompt_table = PromptTemplate.from_template(merge_prompt_template_table)
    logging.info("Merge Prompt templates created.")
    return merge_prompt_default, merge_prompt_table
