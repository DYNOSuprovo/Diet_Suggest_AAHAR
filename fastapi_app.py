# fastapi_app.py
import os
import logging
import random
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

# Google Sheets logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_secret_key_for_fastapi_sessions_change_this")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env or Render secrets!")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)

# --- Google Sheets Setup ---
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "diet-suggest-logger-6e048d507460.json", scope
)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("Diet Suggest Logs").sheet1  # Make sure this sheet exists & is shared

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions.",
    version="0.1.0",
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Local Module Imports ---
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

# --- Initialize LLM and Vector DB ---
llm_gemini = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

try:
    db, _ = setup_vector_database()
    logging.info("✅ Vector DB initialized.")
except Exception as e:
    logging.error(f"Vector DB setup failed: {e}")
    raise HTTPException(status_code=500, detail="Vector DB init error")

# --- Setup Chains ---
rag_prompt = define_rag_prompt_template()
try:
    qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
except RuntimeError as e:
    logging.error(f"QA Chain init failed: {e}")
    raise HTTPException(status_code=500, detail="LangChain QA setup error")

conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
merge_prompt_default, merge_prompt_table = define_merge_prompt_templates()

# --- Pydantic Request Model ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = None

# --- API Routes ---

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    user_query = chat_request.query
    client_session_id = chat_request.session_id

    # --- Session ID Handling ---
    if client_session_id:
        session_id = client_session_id
        request.session['session_id'] = session_id
    else:
        session_id = request.session.get("session_id", f"session_{os.urandom(8).hex()}")
        request.session['session_id'] = session_id

    logging.info(f"📩 Query received: {user_query} | Session: {session_id}")

    # --- Extract User Parameters ---
    user_params = {
        "dietary_type": extract_diet_preference(user_query),
        "goal": extract_diet_goal(user_query),
        "region": extract_regional_preference(user_query)
    }

    logging.info(f"🔍 Extracted Params: {user_params}")

    # --- Get LangChain History ---
    current_langchain_history = get_session_history(session_id)
    chat_history_messages = current_langchain_history.messages

    # --- RAG Answer ---
    try:
        rag_result = await conversational_qa_chain.ainvoke({
            "query": user_query,
            "chat_history": chat_history_messages,
            "dietary_type": user_params["dietary_type"],
            "goal": user_params["goal"],
            "region": user_params["region"]
        }, config={"configurable": {"session_id": session_id}})

        rag_output = rag_result.get("answer", "No answer from RAG.")
        logging.info(f"📚 RAG Output: {rag_output[:100]}...")
    except Exception as e:
        logging.error(f"RAG chain error: {e}", exc_info=True)
        rag_output = "An error occurred while retrieving diet info."

    # --- Groq Suggestions ---
    groq_suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
    if GROQ_API_KEY:
        try:
            groq_suggestions = cached_groq_answers(
                user_query,
                GROQ_API_KEY,
                user_params["dietary_type"],
                user_params["goal"],
                user_params["region"]
            )
        except Exception as e:
            logging.error(f"Groq fetch error: {e}")
            groq_suggestions = {k: f"Error: {e}" for k in groq_suggestions}

    # --- Merge Suggestions ---
    try:
        merge_prompt = merge_prompt_table if contains_table_request(user_query) else merge_prompt_default
        merge_input = {
            "rag": str(rag_output),
            "llama": str(groq_suggestions.get("llama", "N/A")),
            "mixtral": str(groq_suggestions.get("mixtral", "N/A")),
            "gemma": str(groq_suggestions.get("gemma", "N/A")),
            "dietary_type": user_params["dietary_type"],
            "goal": user_params["goal"],
            "region": user_params["region"]
        }

        final_output = await llm_gemini.ainvoke(merge_prompt.format(**merge_input))
        logging.info("✅ Final output generated successfully.")
    except Exception as e:
        logging.error(f"Merge error: {e}", exc_info=True)
        final_output = "Error generating merged response."

    # --- Update History ---
    current_langchain_history.add_user_message(user_query)
    current_langchain_history.add_ai_message(final_output)

    # --- Log to Google Sheets ---
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([
            now,
            session_id,
            user_query,
            final_output
        ])
        logging.info("📝 Logged to Google Sheets.")
    except Exception as log_err:
        logging.error(f"❌ Failed to log to Google Sheet: {log_err}")

    return JSONResponse(content={"answer": final_output, "session_id": session_id})


@app.get("/")
async def root():
    return {"message": "✅ Diet Recommendation API is running. Use POST /chat to interact."}

# --- Render-friendly Port Binding ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=port)
