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

# Google Sheets logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Download & Extract Vector DB ---
def download_and_extract_db():
    url = "https://huggingface.co/datasets/Dyno1307/chromadb-diet/resolve/main/db.zip"
    zip_path = "/tmp/db.zip"
    extract_path = "/tmp/chroma_db"

    try:
        print("⬇️ Downloading Chroma DB zip...")
        response = requests.get(url)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

        print("📦 Extracting zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print("✅ Vector DB extracted.")
    except Exception as e:
        print("❌ Error downloading/extracting DB:", e)
        raise HTTPException(status_code=500, detail="Failed to prepare Vector DB.")

# --- Environment Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_secret_key_for_fastapi_sessions_change_this")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env or Render secrets!")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)

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
        sheet = gs_client.open("Diet Suggest Logs").sheet1
        sheet_enabled = True
        logging.info("✅ Google Sheets connected.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheets disabled: {e}")
else:
    logging.info("ℹ️ GOOGLE_CREDS_JSON not set. Skipping sheet logging.")

# --- FastAPI App Init ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions.",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Import Local Modules ---
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
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

download_and_extract_db()

try:
    db, _ = setup_vector_database(chroma_db_directory="/tmp/chroma_db")
    count = len(db.get()['documents'])
    logging.info(f"✅ Vector DB initialized with {count} documents.")
except Exception as e:
    logging.error(f"❌ Vector DB init error: {e}")
    raise HTTPException(status_code=500, detail="Vector DB init failed")

rag_prompt = define_rag_prompt_template()

try:
    qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
except Exception as e:
    logging.error(f"❌ QA Chain error: {e}")
    raise HTTPException(status_code=500, detail="QA Chain init failed")

conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
merge_prompt_default, merge_prompt_table = define_merge_prompt_templates()

# --- Pydantic Schema ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = None

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    user_query = chat_request.query
    client_session_id = chat_request.session_id
    session_id = client_session_id or request.session.get("session_id") or f"session_{os.urandom(8).hex()}"
    request.session["session_id"] = session_id

    logging.info(f"📩 Query: {user_query} | Session: {session_id}")

    user_params = {
        "dietary_type": extract_diet_preference(user_query),
        "goal": extract_diet_goal(user_query),
        "region": extract_regional_preference(user_query)
    }
    logging.info(f"🔍 Extracted: {user_params}")

    chat_history = get_session_history(session_id).messages
    rag_output = "No answer from RAG."

    try:
        rag_result = await conversational_qa_chain.ainvoke({
            "query": user_query,
            "chat_history": chat_history,
            **user_params
        }, config={
            "callbacks": [SafeTracer()],
            "configurable": {"session_id": session_id}
        })
        rag_output = rag_result
        logging.info(f"✅ RAG Chain Raw Output: {str(rag_output)[:100]}...")
    except Exception as e:
        logging.error(f"❌ RAG error: {e}", exc_info=True)
        rag_output = "Error while retrieving response from knowledge base."

    try:
        groq_suggestions = cached_groq_answers(
            query=user_query,
            groq_api_key=GROQ_API_KEY,
            dietary_type=user_params["dietary_type"],
            goal=user_params["goal"],
            region=user_params["region"]
        )
    except Exception as e:
        logging.error(f"❌ Groq error: {e}")
        groq_suggestions = {"llama": "Error", "mixtral": "Error", "gemma": "Error"}

    try:
        merge_prompt = merge_prompt_table if contains_table_request(user_query) else merge_prompt_default

        final_output = await llm_gemini.ainvoke(
            merge_prompt.format(
                rag_section=f"Primary RAG Answer:\n{rag_output}",
                additional_suggestions_section=(
                    f"- LLaMA Suggestion: {groq_suggestions.get('llama', 'N/A')}\n"
                    f"- Mixtral Suggestion: {groq_suggestions.get('mixtral', 'N/A')}\n"
                    f"- Gemma Suggestion: {groq_suggestions.get('gemma', 'N/A')}"
                ),
                **user_params
            ),
            config={
                "callbacks": [SafeTracer()],
                "configurable": {"session_id": session_id}
            }
        )
    except Exception as e:
        logging.error(f"❌ Merge error: {e}", exc_info=True)
        final_output = "Something went wrong while combining AI suggestions."

    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(final_output)

    try:
        if sheet_enabled and sheet:
            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_id,
                user_query,
                final_output
            ])
            logging.info("📝 Logged to Google Sheet.")
    except Exception as e:
        logging.warning(f"⚠️ Sheet logging failed: {e}")

    return JSONResponse(content={"answer": final_output, "session_id": session_id})

@app.get("/")
async def root():
    return {"message": "✅ Diet Recommendation API is running. Use POST /chat to interact."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
