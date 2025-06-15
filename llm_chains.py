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

# In-memory store for session histories. In a production environment,
# this would typically be replaced by a persistent store (e.g., Redis, database).
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Retrieves or creates a chat message history for a given session ID.
    This helps maintain conversational context across multiple turns.
    """
    if session_id not in store:
        logging.info(f"Creating new Langchain session history in 'store' for: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history from 'store' for: {session_id}")
    return store[session_id]

def define_rag_prompt_template():
    """
    Defines the prompt template for the Retrieval-Augmented Generation (RAG) chain.
    This prompt guides the primary LLM (Gemini) to generate diet suggestions
    based on retrieved context and chat history, tailored to user's parameters.
    """
    template_string = """
    You are AAHAR, an AI assistant specialized in Indian diet and nutrition, created by Suprovo Mallick.
    Your main goal is to provide helpful and culturally relevant Indian diet suggestions.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    User Query: {query}
    Dietary Type Preference: {dietary_type}
    Goal: {goal}
    Region Preference: {region}
    Disease Condition: {disease} # Added disease to the prompt variables if available

    Instructions:
    1. If the user query is clearly related to food, diet, health goals, or nutrition, provide a culturally relevant Indian food suggestion or diet plan.
    2. Tailor your answer for **{dietary_type}** users aiming for **{goal}**, considering the **{region}** Indian context and any **{disease}** conditions mentioned.
    3. Use the provided "Context from Knowledge Base" to ground your answer and ensure accuracy and relevance.
    4. Incorporate "Chat History" to maintain conversational flow and context.
    5. If the user query is a general conversation (like greetings, asking who created you, your name, or what you do), respond politely and DO NOT provide a diet suggestion. These types of queries should ideally be handled by the 'generic_prompt' in the FastAPI app, but this instruction serves as a safeguard.
    6. Ensure the output is actionable and easy to understand.

    Output:
    """
    return PromptTemplate(
        template=template_string,
        # Ensure all input variables match what's passed from FastAPI and the chain
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "disease", "context"]
    )

def define_generic_prompt():
    """
    Defines a prompt template for handling generic, non-diet-specific user queries.
    This is used by FastAPI for quick, polite, and informative responses to greetings
    or meta-questions about the AI.
    """
    template_string = """
    You are AAHAR, an AI assistant specialized in Indian diet and nutrition. You were created by Suprovo Mallick.
    The user's query is '{query}'. This query is not a direct request for a diet plan but a general question or greeting.

    Respond briefly and politely. You can state your purpose (providing Indian diet suggestions) and offer to help with their dietary goals or preferences. Do not generate a diet plan unless explicitly asked.

    Example responses:
    - "Namaste! I'm AAHAR, your AI assistant for healthy Indian diet suggestions. How can I help you today?"
    - "Hello! I am AAHAR, an AI designed to provide personalized Indian diet plans. What kind of diet advice are you looking for?"
    - "Hi there! I am an AI diet recommender. If you tell me your dietary goals or preferences, I can suggest a plan for you."

    Your response:
    """
    return PromptTemplate(
        template=template_string,
        input_variables=["query"]
    )


def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    """
    Sets up the core Retrieval-Augmented Generation (RAG) chain.
    This chain retrieves relevant documents from the vector database and
    uses them to inform the LLM's response.
    """
    try:
        # Retriever configured to fetch top 5 most relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 5})

        def retrieve_and_log_context(input_dict):
            """
            Custom function to retrieve documents and log the context for debugging.
            Ensures that the context passed to the LLM is well-formed.
            """
            docs = retriever.invoke(input_dict["query"])
            if not docs:
                logging.warning(f"No documents retrieved for query: '{input_dict['query']}'")
            # Join the page content of retrieved documents to form the context string
            context_str = "\n\n".join(doc.page_content for doc in docs)
            logging.info(f"Retrieved Context: {context_str[:500]}...") # Log first 500 chars of context
            return context_str

        # Define the chain using LangChain's Runnable API
        # This creates a dictionary of inputs required by the prompt
        qa_chain = (
            {
                "context": retrieve_and_log_context, # The context comes from retrieval
                "query": RunnablePassthrough(),      # User's query passes through
                "chat_history": RunnablePassthrough(), # Chat history passes through
                "dietary_type": RunnablePassthrough(), # Dietary type passes through
                "goal": RunnablePassthrough(),       # Goal passes through
                "region": RunnablePassthrough(),     # Region passes through
                "disease": RunnablePassthrough(),    # Disease passes through
            }
            | rag_prompt       # The prompt takes these inputs
            | llm_gemini       # The LLM generates a response
            | StrOutputParser() # The output is parsed as a string
        )
        logging.info("Retrieval QA Chain initialized successfully.")
        return qa_chain
    except Exception as e:
        logging.exception("Full QA Chain setup traceback:") # Log full traceback for better debugging
        raise RuntimeError(f"QA Chain setup error: {e}")


def setup_conversational_qa_chain(qa_chain):
    """
    Wraps the QA chain with conversational history management.
    This allows the AI to remember previous turns in the conversation.
    """
    conversational_qa_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="query",        # Key for the user's current message
        history_messages_key="chat_history", # Key for the full chat history
        # output_messages_key="answer" # Not needed for StrOutputParser, as it's directly returned
    )
    logging.info("Conversational QA Chain initialized.")
    return conversational_qa_chain


def define_merge_prompt_templates():
    """
    Defines multiple prompt templates for merging RAG and Groq outputs
    into a final, coherent response, supporting different formatting requests.
    """
    # Default merge prompt for general responses
    merge_prompt_default_template = """
    You are an AI assistant specializing in Indian diet and nutrition, created by Suprovo Mallick.
    Your task is to provide a single, coherent, and practical **{dietary_type}** food suggestion or diet plan for **{goal}**,
    tailored for a **{region}** Indian context. If a disease condition **{disease_section}** is specified, incorporate relevant considerations.
    The user's original query was: "{query}"

    Here's the information available:
    Primary RAG Answer:\n{rag_section}
    Additional AI Suggestions (from other LLMs for diverse ideas):\n{additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional AI Suggestions".
    3. Combine information logically and seamlessly, without explicitly mentioning the source of each piece (e.g., "from RAG," "Groq said").
    4. Ensure the final plan is clear, actionable, culturally relevant, and addresses the user's stated dietary type, goal, region, and disease condition.
    5. Maintain a helpful and professional tone.

    Final {dietary_type} {goal} Food Suggestion/Diet Plan (Tailored for {region} Indian context):
    """

    # Merge prompt for generating output in table format
    merge_prompt_table_template = """
    You are an AI assistant specializing in Indian diet and nutrition, created by Suprovo Mallick.
    Your task is to provide a single, coherent, and practical **{dietary_type}** food suggestion or diet plan for **{goal}**,
    tailored for a **{region}** Indian context. If a disease condition **{disease_section}** is specified, incorporate relevant considerations.
    **You MUST present the final diet plan as a clear markdown table.**
    Include columns for "Meal", "Food Items", and "Notes/Considerations".
    The user's original query was: "{query}"

    Here's the information available:
    Primary RAG Answer:\n{rag_section}
    Additional AI Suggestions (from other LLMs for diverse ideas):\n{additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional AI Suggestions".
    3. Combine information logically and seamlessly, without explicitly mentioning the source of each piece.
    4. Ensure the final plan is clear, actionable, and culturally relevant.
    5. **Strictly adhere to markdown table format.**
    6. Maintain a helpful and professional tone.

    Final {dietary_type} {goal} Diet Plan (Tailored for {region} Indian context, in markdown table format):
    """

    # Merge prompt for generating output in paragraph format
    merge_prompt_paragraph_template = """
    You are an AI assistant specializing in Indian diet and nutrition, created by Suprovo Mallick.
    Your task is to provide a single, coherent, and practical **{dietary_type}** food suggestion or diet plan for **{goal}**,
    tailored for a **{region}** Indian context. If a disease condition **{disease_section}** is specified, incorporate relevant considerations.
    **You MUST present the final diet plan as a detailed and well-structured paragraph (or continuous prose).**
    Organize the information logically, perhaps by meal times or aspects of the diet.
    The user's original query was: "{query}"

    Here's the information available:
    Primary RAG Answer:\n{rag_section}
    Additional AI Suggestions (from other LLMs for diverse ideas):\n{additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional AI Suggestions".
    3. Combine information logically and seamlessly, without explicitly mentioning the source of each piece.
    4. Ensure the final plan is clear, actionable, and culturally relevant.
    5. **Strictly present the diet plan as continuous paragraphs.**
    6. Maintain a helpful and professional tone.

    Final {dietary_type} {goal} Diet Plan (Tailored for {region} Indian context, in paragraph form):
    """

    # Return all three prompt templates
    logging.info("Merge Prompt templates created.")
    return (
        PromptTemplate(template=merge_prompt_default_template, input_variables=["rag_section", "additional_suggestions_section", "query", "dietary_type", "goal", "region", "disease_section"]),
        PromptTemplate(template=merge_prompt_table_template, input_variables=["rag_section", "additional_suggestions_section", "query", "dietary_type", "goal", "region", "disease_section"]),
        PromptTemplate(template=merge_prompt_paragraph_template, input_variables=["rag_section", "additional_suggestions_section", "query", "dietary_type", "goal", "region", "disease_section"])
    )
