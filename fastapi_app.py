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
# Import AIMessage to explicitly handle LLM outputs
from langchain_core.messages import AIMessage 

# Google Sheets logging (Ensure gspread and oauth2client are installed: pip install gspread oauth2client)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Download & Extract Vector DB ---
def download_and_extract_db():
    url = "https://huggingface.co/datasets/Dyno1307/chromadb-diet/resolve/main/db.zip"
    zip_path = "/tmp/db.zip"
    extract_path = "/tmp/chroma_db" # MUST match embedding_utils.py

    # Check if DB already exists to avoid re-downloading on every startup
    if os.path.exists(os.path.join(extract_path, "index")):
        logging.info("✅ Chroma DB already exists, skipping download.")
        return

    try:
        logging.info("⬇️ Downloading Chroma DB zip from HuggingFace...")
        response = requests.get(url, timeout=60) # Added timeout for robustness
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
# Import the enhanced query analysis functions from query_analysis
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

# --- LLM & Vector DB Setup ---
llm_gemini = GoogleGenerativeAI(
    model="gemini-1.5-flash", # Using the faster flash model
    google_api_key=GEMINI_API_KEY,
    temperature=0.5 # A moderate temperature for balanced creativity and factualness
)

# Initialize generic response LLM (can be a lighter model if preferred)
llm_generic = GoogleGenerativeAI( # NOTE: This was not in your old code, but it's a good separation of concerns
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
    logging.error(f"❌ Vector DB init error: {e}", exc_info=True)
    # If DB fails, the app might not function correctly for RAG queries
    raise HTTPException(status_code=500, detail="Vector DB init failed")

# Setup LangChain components
rag_prompt = define_rag_prompt_template()

qa_chain = None
conversational_qa_chain = None
try:
    qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
    conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
    logging.info("✅ QA Chains initialized successfully.")
except Exception as e:
    logging.error(f"❌ QA Chain setup error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="QA Chain initialization failed")

merge_prompt_default, merge_prompt_table = define_merge_prompt_templates() # Only two merge prompts in old code

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

    logging.info(f"📩 Query: '{user_query}' | Session: {session_id}")

    # Use older query analysis functions
    is_greeting_flag = is_greeting(user_query)
    is_formatting_flag = is_formatting_request(user_query)
    wants_table_flag = contains_table_request(user_query)
    
    # Use older param extraction
    user_params = {
        "dietary_type": extract_diet_preference(user_query),
        "goal": extract_diet_goal(user_query),
        "region": extract_regional_preference(user_query)
    }
    logging.info(f"🔍 Extracted: {user_params} | Greeting: {is_greeting_flag} | Formatting: {is_formatting_flag} | Table: {wants_table_flag}")

    chat_history = get_session_history(session_id).messages # Retrieve chat history for this session

    # --- Intent-Based Routing (adapted from your older code logic) ---
    response_text = ""

    if is_greeting_flag:
        response_text = "Namaste! How can I assist you with a healthy Indian diet today?"
    elif is_formatting_flag and len(chat_history) > 1: # Check if there's previous AI message to reformat
        # This part of the logic needs to be more robust, as in the newer versions.
        # For now, we'll keep it simple as in your provided old code structure.
        # This will need refinement for specific reformatting of previous AI output.
        logging.info("Handling formatting request (simple).")
        last_ai_message_content = None
        for msg in reversed(chat_history):
            if hasattr(msg, 'content') and msg.content and len(msg.content) > 50: # Simple heuristic
                last_ai_message_content = msg.content
                break

        if last_ai_message_content:
            merge_prompt = merge_prompt_table if wants_table_flag else merge_prompt_default
            try:
                final_output_obj = await llm_gemini.ainvoke(
                    merge_prompt.format(
                        rag_section=f"Previous Answer:\n{last_ai_message_content}",
                        additional_suggestions_section="No new suggestions needed for reformatting.",
                        query=user_query, # Pass user query for context if needed in prompt
                        **user_params # Pass extracted params for prompt context
                    ),
                    config={
                        "callbacks": [SafeTracer()],
                        "configurable": {"session_id": session_id}
                    }
                )
                response_text = final_output_obj.content
            except Exception as e:
                logging.error(f"❌ Reformat error: {e}", exc_info=True)
                response_text = "I encountered an issue reformatting the previous response. Please try again."
        else:
            response_text = "I can only re-format a previous diet plan. Please ask for a diet plan first!"

    else: # Default to RAG + Groq + Merge pipeline for task-oriented queries or general questions
        logging.info("Handling RAG + Groq + Merge pipeline.")
        rag_output_content = "No answer from RAG."
        try:
            rag_result = await conversational_qa_chain.ainvoke({
                "query": user_query,
                "chat_history": chat_history,
                **user_params
            }, config={
                "callbacks": [SafeTracer()],
                "configurable": {"session_id": session_id}
            })
            # In your older llm_chains.py, qa_chain (and thus conversational_qa_chain)
            # ends with StrOutputParser(), so rag_result should be a string directly.
            rag_output_content = str(rag_result) # Explicitly cast to string
            logging.info(f"✅ RAG Chain Raw Output: {rag_output_content[:200]}...")
        except Exception as e:
            logging.error(f"❌ RAG error during ainvoke: {e}", exc_info=True)
            rag_output_content = "Error while retrieving response from knowledge base."

        groq_suggestions = {}
        try:
            groq_suggestions = cached_groq_answers(
                query=user_query,
                groq_api_key=GROQ_API_KEY,
                dietary_type=user_params["dietary_type"],
                goal=user_params["goal"],
                region=user_params["region"]
            )
        except Exception as e:
            logging.error(f"❌ Groq error during suggestions: {e}", exc_info=True)
            groq_suggestions = {"llama": "Error", "mixtral": "Error", "gemma": "Error"}

        merge_prompt = merge_prompt_table if wants_table_flag else merge_prompt_default

        final_output_content = "Something went wrong while combining AI suggestions."
        try:
            format_kwargs = {
                "rag_section": f"Primary RAG Answer:\n{rag_output_content}",
                "additional_suggestions_section": (
                    f"- LLaMA Suggestion: {groq_suggestions.get('llama', 'N/A')}\n"
                    f"- Mixtral Suggestion: {groq_suggestions.get('mixtral', 'N/A')}\n"
                    f"- Gemma Suggestion: {groq_suggestions.get('gemma', 'N/A')}"
                ),
                "query": user_query,
                "dietary_type": user_params.get("dietary_type", "any"),
                "goal": user_params.get("goal", "general"),
                "region": user_params.get("region", "Indian"),
                "disease_section": "" # Older code didn't have disease, so ensure it's empty
            }
            # The older merge prompts only had "rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"
            # It's crucial to match the input_variables of the prompt template here.
            # Assuming the old merge prompts had {query} too from the previous context.
            # Let's adjust format_kwargs to strictly match the older merge prompt's variables,
            # which are less comprehensive than the ones I was using.
            
            # Re-checking the older prompt template from your input:
            # input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"]
            # No "query" or "disease_section" in older merge prompts.
            # This is a key finding!

            # ADJUSTED format_kwargs to match the old merge prompt's expected variables
            strict_old_format_kwargs = {
                "rag_section": f"Primary RAG Answer:\n{rag_output_content}",
                "additional_suggestions_section": (
                    f"- LLaMA Suggestion: {groq_suggestions.get('llama', 'N/A')}\n"
                    f"- Mixtral Suggestion: {groq_suggestions.get('mixtral', 'N/A')}\n"
                    f"- Gemma Suggestion: {groq_suggestions.get('gemma', 'N/A')}"
                ),
                "dietary_type": user_params.get("dietary_type", "any"),
                "goal": user_params.get("goal", "general"),
                "region": user_params.get("region", "Indian")
            }

            logging.info(f"➡️ Sending Merge Prompt (length before format: {len(merge_prompt.template)} chars) to LLM for final output.")
            merge_result_obj = await llm_gemini.ainvoke(
                merge_prompt.format(**strict_old_format_kwargs), # Use strict_old_format_kwargs
                config={
                    "callbacks": [SafeTracer()],
                    "configurable": {"session_id": session_id}
                }
            )
            final_output_content = merge_result_obj.content # Assuming AIMessage.content as before
        except Exception as e:
            logging.error(f"❌ Merge process error: {type(e).__name__}: {e}", exc_info=True)
            final_output_content = "I encountered an issue generating a comprehensive diet plan. Please try again."
        
        response_text = final_output_content


    # Add user and AI messages to session history (this was at the very end in your old code)
    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(response_text) # Use response_text consistently

    # Log to Google Sheet (this was at the very end in your old code)
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
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000))) # Changed default to 10000 per your old code
