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
from langchain_core.messages import AIMessage, HumanMessage # Import message types

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
    """
    Defines the prompt template for the Retrieval-Augmented Generation (RAG) chain.
    This prompt guides the primary LLM (Gemini) to generate diet suggestions
    based on retrieved context and chat history, tailored to user's parameters.
    Removed instruction for handling general conversation as this is done by FastAPI routing.
    """
    template_string = """
    You are AAHAR, an AI assistant specialized in Indian diet and nutrition, created by Suprovo Mallick.
    Your task is to provide a culturally relevant Indian food suggestion or diet plan.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    User Query: {query}
    Dietary Type Preference: {dietary_type}
    Goal: {goal}
    Region Preference: {region}
    Disease Condition: {disease}

    Instructions:
    1. Provide a clear, actionable, and culturally relevant Indian food suggestion or diet plan based on the user's query, dietary type, goal, region, and any disease conditions.
    2. Prioritize and synthesize information from the "Context from Knowledge Base" and "Chat History".
    3. If the context is insufficient or irrelevant for the diet request, state that you cannot provide a detailed plan and suggest trying another query.
    4. Focus solely on providing the diet suggestion; avoid conversational pleasantries (these are handled upstream).

    Output:
    """
    return PromptTemplate(
        template=template_string,
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
    - "Hi there! I am an AI diet recommender. If you tell me my dietary goals or preferences, I can suggest a plan for you."

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
        retriever = db.as_retriever(search_kwargs={"k": 5})

        def retrieve_and_log_context(input_dict):
            docs = retriever.invoke(input_dict["query"])
            if not docs:
                logging.warning(f"No documents retrieved for query: '{input_dict['query']}'")
            context_str = "\n\n".join(doc.page_content for doc in docs)
            logging.info(f"Retrieved Context: {context_str[:500]}...")
            return context_str

        # The qa_chain will produce an AIMessage if StrOutputParser() is NOT the last step.
        # But if it is, it produces a string. For conversational_qa_chain to properly
        # extract the 'answer', the wrapped chain needs to return a dictionary with 'answer' key.
        # OR, we need to explicitly extract .content from the AIMessage if we keep StrOutputParser later.
        # Let's make this return AIMessage, and let StrOutputParser be handled outside if needed.

        # MODIFIED: Removed StrOutputParser here, so this chain returns an AIMessage.
        # The conversational_qa_chain will handle output_messages_key="answer" based on this.
        qa_chain = (
            {
                "context": retrieve_and_log_context,
                "query": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "dietary_type": RunnablePassthrough(),
                "goal": RunnablePassthrough(),
                "region": RunnablePassthrough(),
                "disease": RunnablePassthrough(),
            }
            | rag_prompt
            | llm_gemini # This will now return an AIMessage
        )
        logging.info("Retrieval QA Chain initialized successfully (returns AIMessage).")
        return qa_chain
    except Exception as e:
        logging.exception("Full QA Chain setup traceback:")
        raise RuntimeError(f"QA Chain setup error: {e}")


def setup_conversational_qa_chain(qa_chain):
    """
    Wraps the QA chain with conversational history management.
    This allows the AI to remember previous turns in the conversation.
    MODIFIED: Added output_messages_key="answer" to explicitly tell RunnableWithMessageHistory
    which part of the wrapped chain's output is the answer to be stored/returned.
    """
    conversational_qa_chain = RunnableWithMessageHistory(
        qa_chain, # This qa_chain now returns an AIMessage
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer" # <--- IMPORTANT: Re-enabled and ensures 'answer' key in output
    )
    logging.info("Conversational QA Chain initialized (with output_messages_key).")
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
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
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
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
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
