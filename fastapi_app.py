# fastapi_app.py

import os
import json
import logging
import zipfile
import requests
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from custom_callbacks import SafeTracer

# Google Sheets logging (Ensure gspread and oauth2client are installed: pip install gspread oauth2client)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Download & Extract Vector DB ---
def download_and_extract_db():
    url = "https://huggingface.co/datasets/Dyno1307/chromadb-diet/resolve/main/db.zip"
    zip_path = "/tmp/db.zip"
    extract_path = "/tmp/chroma_db"

    # Skip if already extracted (check for a representative file, e.g., 'index' directory)
    if os.path.exists(os.path.join(extract_path, "index")):
        logging.info("✅ Chroma DB already exists, skipping download.")
        return

    try:
        logging.info("⬇️ Downloading Chroma DB zip from HuggingFace...")
        response = requests.get(url, timeout=60) # Increased timeout for larger files
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        with open(zip_path, "wb") as f:
            f.write(response.content)

        logging.info("📦 Extracting zip to /tmp...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        logging.info("✅ Vector DB extracted successfully.")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Network or HTTP error downloading Vector DB: {req_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download Vector DB: {req_err}")
    except zipfile.BadZipFile:
        logging.error("❌ Downloaded file is not a valid zip file.", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to extract Vector DB: Corrupted zip file.")
    except Exception as e:
        logging.error(f"❌ General error downloading or extracting Vector DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to prepare Vector DB: {e}")

# --- Environment Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_secret_key_for_fastapi_sessions_change_this")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON") # This should be a JSON string from your service account key

if not GEMINI_API_KEY:
    logging.critical("❌ GEMINI_API_KEY not found in .env or environment variables!")
    # Depending on deployment, you might want to raise an exception or run in a limited mode
    raise ValueError("GEMINI_API_KEY not set. Please set it in your .env file or environment.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy logs from specific langchain modules
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING) # Suppress http client logs

# --- Google Sheets Setup ---
sheet = None
sheet_enabled = False
if GOOGLE_CREDS_JSON:
    try:
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive",
             "https://www.googleapis.com/auth/spreadsheets"]
        )
        gs_client = gspread.authorize(creds)
        # Replace "Diet Suggest Logs" with your actual Google Sheet name
        sheet = gs_client.open("Diet Suggest Logs").sheet1 
        sheet_enabled = True
        logging.info("✅ Google Sheets connected for logging.")
    except json.JSONDecodeError:
        logging.warning("⚠️ GOOGLE_CREDS_JSON is not a valid JSON string. Google Sheets disabled.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheets connection failed: {e}. Logging to sheet disabled.", exc_info=True)
else:
    logging.info("ℹ️ GOOGLE_CREDS_JSON not set. Skipping Google Sheets logging.")

# --- FastAPI App Init ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions using RAG and LLMs.",
    version="0.2.0", # Increment version as changes are made
)
# CORS configuration for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this to specific origins in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Session middleware for conversational history
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Import Local Modules ---
# Import the enhanced extract_all_metadata from query_analysis
from query_analysis import extract_all_metadata 

from llm_chains import (
    define_rag_prompt_template, setup_qa_chain, setup_conversational_qa_chain,
    define_merge_prompt_templates, get_session_history, define_generic_prompt
)
from embedding_utils import setup_vector_database
from groq_integration import cached_groq_answers

# --- LLM & Vector DB Setup ---
llm_gemini = GoogleGenerativeAI(
    model="gemini-1.5-flash", # Using the faster flash model
    google_api_key=GEMINI_API_KEY,
    temperature=0.5 # A moderate temperature for balanced creativity and factualness
)

# Initialize generic response LLM (can be a lighter model if preferred)
llm_generic = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.7 # Slightly higher temp for more conversational replies
)

# Download and extract the database at app startup
download_and_extract_db()

# Setup Vector DB
db = None
try:
    db, _ = setup_vector_database(chroma_db_directory="/tmp/chroma_db")
    count = len(db.get()['documents'])
    logging.info(f"✅ Vector DB initialized with {count} documents.")
except Exception as e:
    logging.error(f"❌ Vector DB initialization error: {e}", exc_info=True)
    # If DB fails, the app might not function correctly for RAG queries
    raise HTTPException(status_code=500, detail="Vector DB initialization failed. Check logs for details.")

# Setup LangChain components
rag_prompt = define_rag_prompt_template()
generic_prompt = define_generic_prompt() # Define a specific prompt for generic queries

qa_chain = None
conversational_qa_chain = None
try:
    qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
    conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
    logging.info("✅ QA Chains initialized successfully.")
except Exception as e:
    logging.error(f"❌ QA Chain setup error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="QA Chain initialization failed. Check logs for details.")

merge_prompt_default, merge_prompt_table, merge_prompt_paragraph = define_merge_prompt_templates()

# --- Pydantic Schema for Request Body ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = None # Optional session ID from client

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    user_query = chat_request.query
    client_session_id = chat_request.session_id
    
    # Generate or retrieve session ID for conversational memory
    session_id = client_session_id or request.session.get("session_id") or f"session_{os.urandom(8).hex()}"
    request.session["session_id"] = session_id # Store in FastAPI session for next request

    logging.info(f"📩 Incoming Query: '{user_query}' | Session: {session_id}")

    # Use the enhanced query_analysis to get all metadata
    query_metadata = extract_all_metadata(user_query)
    logging.info(f"🔍 Query Metadata: {query_metadata}")

    response_text = ""
    chat_history = get_session_history(session_id).messages # Retrieve chat history for this session

    try:
        # --- Intent-Based Routing ---
        if query_metadata["primary_intent_type"] == "greeting":
            response_text = "Namaste! How can I assist you with a healthy Indian diet today?"
        
        elif query_metadata["primary_intent_type"] == "generic":
            # Use a simpler LLM interaction for generic questions
            logging.info("Handling generic query.")
            generic_response_content = await llm_generic.ainvoke(
                generic_prompt.format(query=user_query),
                config={"callbacks": [SafeTracer()], "configurable": {"session_id": session_id}}
            )
            response_text = generic_response_content.content # Extract content from AIMessage
            
        elif query_metadata["primary_intent_type"] == "formatting" and query_metadata["is_follow_up"]:
            # This is a request to re-format a previous answer.
            # Requires storing the *last generated diet plan* in session memory.
            # For this example, we'll give a placeholder or re-generate if no memory.
            last_ai_message = next((msg.content for msg in reversed(chat_history) if msg.type == 'ai'), None)
            if last_ai_message:
                logging.info("Reformatting previous response.")
                # Use a specific merge prompt to reformat the last AI message
                if query_metadata["wants_table"]:
                    merge_prompt_for_reformat = merge_prompt_table
                elif query_metadata["wants_paragraph"]:
                    merge_prompt_for_reformat = merge_prompt_paragraph
                else: # Default to general merge if specific format not found but it's formatting
                    merge_prompt_for_reformat = merge_prompt_default

                reformatted_content = await llm_gemini.ainvoke(
                    merge_prompt_for_reformat.format(
                        rag_section=f"Previous Answer:\n{last_ai_message}",
                        additional_suggestions_section="No new suggestions needed for reformatting.", # Can be empty
                        dietary_type="", goal="", region="" # Pass empty as context is the previous answer
                    ),
                    config={"callbacks": [SafeTracer()], "configurable": {"session_id": session_id}}
                )
                response_text = reformatted_content.content
            else:
                response_text = "I can only re-format a previous diet plan. Please ask for a diet plan first!"
                
        else: # primary_intent_type is "task" or a formatting request with task keywords
            # Proceed with the full RAG and merge pipeline for task-oriented queries
            logging.info("Handling task-oriented query (RAG + Groq + Merge).")
            
            user_params = { # Use extracted params for the RAG chain
                "dietary_type": query_metadata["dietary_type"],
                "goal": query_metadata["goal"],
                "region": query_metadata["region"],
                "disease": query_metadata["disease"] # Pass disease info if available
            }

            rag_output = "No answer from RAG."
            try:
                rag_result = await conversational_qa_chain.ainvoke({
                    "query": user_query,
                    "chat_history": chat_history,
                    **user_params # Pass extracted user parameters to the chain
                }, config={
                    "callbacks": [SafeTracer()],
                    "configurable": {"session_id": session_id}
                })
                # Check if rag_result is a dict with an 'answer' key or just the content
                rag_output = rag_result.get("answer", str(rag_result)) if isinstance(rag_result, dict) else str(rag_result)
                logging.info(f"✅ RAG Chain Raw Output: {rag_output[:200]}...") # Log first 200 chars
            except Exception as e:
                logging.error(f"❌ RAG error during ainvoke: {e}", exc_info=True)
                rag_output = "Error while retrieving response from knowledge base."

            groq_suggestions = {}
            try:
                groq_suggestions = cached_groq_answers(
                    query=user_query,
                    groq_api_key=GROQ_API_KEY,
                    # Pass the extracted params to Groq for contextual suggestions
                    dietary_type=user_params["dietary_type"],
                    goal=user_params["goal"],
                    region=user_params["region"]
                )
            except Exception as e:
                logging.error(f"❌ Groq error during suggestions: {e}", exc_info=True)
                groq_suggestions = {"llama": "Error", "mixtral": "Error", "gemma": "Error"}

            # Select merge prompt based on formatting request
            if query_metadata["wants_table"]:
                merge_prompt = merge_prompt_table
            elif query_metadata["wants_paragraph"]:
                merge_prompt = merge_prompt_paragraph
            else:
                merge_prompt = merge_prompt_default

            final_output_content = "Something went wrong while combining AI suggestions."
            try:
                # Use a dict for format_kwargs to handle optional disease parameter
                format_kwargs = {
                    "rag_section": f"Primary RAG Answer:\n{rag_output}",
                    "additional_suggestions_section": (
                        f"- LLaMA Suggestion: {groq_suggestions.get('llama', 'N/A')}\n"
                        f"- Mixtral Suggestion: {groq_suggestions.get('mixtral', 'N/A')}\n"
                        f"- Gemma Suggestion: {groq_suggestions.get('gemma', 'N/A')}"
                    ),
                    "dietary_type": user_params["dietary_type"],
                    "goal": user_params["goal"],
                    "region": user_params["region"],
                    "query": user_query # Include original query in merge prompt for better context
                }
                if user_params["disease"]:
                    format_kwargs["disease"] = user_params["disease"] # Add disease if present

                # It's crucial that your merge_prompt templates can handle all these format_kwargs
                # Define merge_prompt_templates needs to be updated to take `disease` if used.
                merge_result = await llm_gemini.ainvoke(
                    merge_prompt.format(**format_kwargs),
                    config={
                        "callbacks": [SafeTracer()],
                        "configurable": {"session_id": session_id}
                    }
                )
                final_output_content = merge_result.content # Extract content from AIMessage
            except Exception as e:
                logging.error(f"❌ Merge process error: {e}", exc_info=True)
                final_output_content = "I encountered an issue generating a comprehensive diet plan. Please try again."
            
            response_text = final_output_content

    except Exception as e:
        logging.error(f"❌ Unhandled error in /chat endpoint: {e}", exc_info=True)
        response_text = "I'm sorry, an unexpected error occurred. Please try again."

    # Add user and AI messages to session history
    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(response_text)

    # Log to Google Sheet
    try:
        if sheet_enabled and sheet:
            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_id,
                user_query,
                response_text # Log the actual response sent to the user
            ])
            logging.info("📝 Logged query and response to Google Sheet.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheet logging failed: {e}", exc_info=True)

    # Return the response as JSON
    return JSONResponse(content={"answer": response_text, "session_id": session_id})

@app.get("/")
async def root():
    return {"message": "✅ Indian Diet Recommendation API is running. Use POST /chat to interact."}

if __name__ == "__main__":
    import uvicorn
    # The default port for Render is 10000, but use 8000 for local development if not set
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True) # reload=True for dev
