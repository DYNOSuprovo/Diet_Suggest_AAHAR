# fastapi_app.py (FastAPI Backend Application)
import os
import logging
import random
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Import your local modules
from query_analysis import (
    is_greeting, is_formatting_request, contains_table_request,
    extract_diet_preference, extract_diet_goal, extract_regional_preference
)
from llm_chains import (
    define_rag_prompt_template, setup_qa_chain, setup_conversational_qa_chain,
    define_merge_prompt_templates, get_session_history
)
from embedding_utils import setup_vector_database
from groq_integration import cached_groq_answers

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions.",
    version="0.1.0",
)

# --- DEBUGGING PRINTS (These lines are added to diagnose the Flask loading issue) ---
# These prints will show you what 'app' object Uvicorn is actually loading.
print(f"DEBUGGING APP TYPE: App object type when loaded: {type(app)}")
print(f"DEBUGGING APP MODULE: App object module when loaded: {app.__module__}")
# --- END DEBUGGING PRINTS ---

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Add Session Middleware (for simple session management) ---
# In a real production app, use a more robust session backend (e.g., Redis)
# This uses itsdangerous to sign the session cookie. YOU MUST CHANGE THIS KEY!
app.add_middleware(SessionMiddleware, secret_key=os.getenv("FASTAPI_SECRET_KEY", "a_very_secret_key_for_fastapi_sessions_change_this"))


# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- API Key Checks ---
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    raise ValueError("GEMINI_API_KEY not found.")

if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY not found. Groq suggestions will be unavailable.")


# --- Global LLM and Chain Initialization (happens once when app starts) ---
llm_gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.5)

# Setup Vector Database and Embeddings
try:
    db, _ = setup_vector_database()
    logging.info("Vector database and embeddings set up.")
except Exception as e:
    logging.error(f"Failed to set up vector database: {e}")
    raise HTTPException(status_code=500, detail="Application failed to start due to VectorDB error.")

# Setup Langchain Chains
rag_prompt = define_rag_prompt_template()
try:
    qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
except RuntimeError as e:
    logging.error(f"Failed to set up QA Chain: {e}")
    raise HTTPException(status_code=500, detail="Application failed to start due to QA Chain error.")

conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
merge_prompt_default, merge_prompt_table = define_merge_prompt_templates()
logging.info("All Langchain chains and prompts initialized.")


# --- Pydantic Model for Request Body ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = None # Optional session ID from frontend


# --- API Routes ---

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    """
    Handles chat requests.
    Expects a JSON payload with a 'query' field and optional 'session_id'.
    Returns a JSON response with the 'answer' and the 'session_id'.
    """
    user_query = chat_request.query
    client_session_id = chat_request.session_id

    # Determine session ID for Langchain history
    # Prioritize client_session_id if provided, otherwise use server-side session
    if client_session_id:
        current_session_id = client_session_id
        # Optionally, update the server-side session cookie with the client's ID
        # if you want to ensure consistency across browser refreshes without resending the ID.
        request.session['session_id'] = current_session_id
        logging.info(f"Using session ID from client request: {current_session_id}")
    else:
        # If client didn't provide session_id, check server-side session
        current_session_id = request.session.get('session_id')
        if not current_session_id:
            # If still no session_id, generate a new one and store in server-side session
            current_session_id = "session_" + os.urandom(8).hex()
            request.session['session_id'] = current_session_id
            logging.info(f"New FastAPI session ID generated: {current_session_id}")
        else:
            logging.info(f"Using existing FastAPI session ID from server session: {current_session_id}")


    logging.info(f"Received query: '{user_query}' for session: {current_session_id}")

    # Extract user preferences from the query
    user_params = {
        "dietary_type": extract_diet_preference(user_query),
        "goal": extract_diet_goal(user_query),
        "region": extract_regional_preference(user_query)
    }
    logging.info(f"Extracted User Params: {user_params}")

    # Get current chat history for the session
    current_langchain_history = get_session_history(current_session_id)
    chat_history_messages = current_langchain_history.messages

    # --- RAG Answer (Primary) ---
    rag_output = "Could not retrieve from knowledge base."
    try:
        # Pass 'query' as the input key, aligning with the updated chains
        rag_result = await conversational_qa_chain.ainvoke( # Use ainvoke for async calls
            {"query": user_query,
             "chat_history": chat_history_messages,
             "dietary_type": user_params["dietary_type"],
             "goal": user_params["goal"],
             "region": user_params["region"]},
            config={"configurable": {"session_id": current_session_id}}
        )

        if isinstance(rag_result, dict) and rag_result.get('answer'):
            rag_output = rag_result["answer"]
            logging.info(f"RAG output FOUND: {rag_output[:100]}...")
        else:
            logging.warning("RAG output is EMPTY or 'answer' key is missing/None.")
    except Exception as e:
        logging.error(f"Error invoking RAG chain: {e}", exc_info=True)
        rag_output = "An error occurred while getting the primary RAG answer."

    # --- Other LLM Suggestions (Groq) ---
    groq_suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
    if GROQ_API_KEY:
        try:
            # cached_groq_answers is sync, so no await
            groq_suggestions = cached_groq_answers(
                user_query,
                GROQ_API_KEY,
                user_params["dietary_type"],
                user_params["goal"],
                user_params["region"]
            )
            logging.info("Groq suggestions obtained.")
        except Exception as e:
            logging.error(f"Error getting Groq answers: {e}", exc_info=True)
            groq_suggestions = {k: f"Error: {e}" for k in groq_suggestions}
    else:
        groq_suggestions = {k: "Groq API key not available." for k in groq_suggestions}

    # --- Merge Suggestions ---
    final_response_content = ""
    try:
        if contains_table_request(user_query):
            merge_prompt = merge_prompt_table
        else:
            merge_prompt = merge_prompt_default

        merge_input = {
            "rag": str(rag_output),
            "llama": str(groq_suggestions.get("llama", "N/A")),
            "mixtral": str(groq_suggestions.get("mixtral", "N/A")),
            "gemma": str(groq_suggestions.get("gemma", "N/A")),
            "dietary_type": user_params["dietary_type"],
            "goal": user_params["goal"],
            "region": user_params["region"]
        }
        # Use ainvoke for async LLM calls
        final_response_content = await llm_gemini.ainvoke(
            merge_prompt.format(**merge_input)
        )
        logging.info("Suggestions merged successfully.")

    except Exception as e:
        logging.error(f"Error merging suggestions: {e}", exc_info=True)
        final_response_content = "An error occurred while generating the final response. Please try again."

    # Update Langchain history with user query and final AI response
    current_langchain_history.add_user_message(user_query)
    current_langchain_history.add_ai_message(final_response_content)

    return JSONResponse(content={"answer": final_response_content, "session_id": current_session_id})


@app.get("/")
async def root():
    """
    Root endpoint to confirm the API is running.
    """
    return {"message": "Diet Recommendation FastAPI Backend is running. Send POST requests to /chat"}

