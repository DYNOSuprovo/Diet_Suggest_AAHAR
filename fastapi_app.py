import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
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

# --- Initialize FastAPI ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions.",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Local Imports ---
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

# --- LLM & VectorDB Setup ---
llm_gemini = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

try:
    db, _ = setup_vector_database()
    logging.info("✅ Vector DB initialized.")
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

# --- Schema ---
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

    # --- Start of RAG processing ---
    rag_output = "No answer from RAG." # Default value if RAG fails

    try:
        # Pass all user_params and history to the conversational chain
        rag_result = await conversational_qa_chain.ainvoke({
            "query": user_query,
            "chat_history": chat_history,
            **user_params # This expands dietary_type, goal, region
        }, config={"configurable": {"session_id": session_id}})

        # CRITICAL FIX 1: rag_result is now directly the string output, not a dictionary
        rag_output = rag_result
        logging.info(f"✅ RAG Chain Raw Output: {rag_output[:100]}...") # Log beginning of successful RAG output

    except Exception as e:
        logging.error(f"❌ RAG error: {e}", exc_info=True)
        # Keep this specific message so the merge prompt can conditionally ignore it
        rag_output = "Error while retrieving response from knowledge base."
        # No need to raise HTTPException here, we want to fallback to Groq

    # --- Groq suggestions (remains mostly same) ---
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

    # --- Merging Logic (CRITICAL FIX 2) ---
    try:
        merge_prompt = merge_prompt_table if contains_table_request(user_query) else merge_prompt_default

        # Construct the sections for the merge prompt
        rag_section_content = f"Primary RAG Answer:\n{rag_output}"

        # If RAG explicitly failed, we might want to make the "Primary RAG Answer" section simpler
        # or indicate its failure to the LLM less prominently.
        # However, the prompt itself is now designed to handle "Error while retrieving..." gracefully.
        # So, we just pass the rag_output as is.

        additional_suggestions_section_content = (
            f"Additional Suggestions (for fallback or enhancement):\n"
            f"- LLaMA Suggestion: {groq_suggestions.get('llama', 'N/A')}\n"
            f"- Mixtral Suggestion: {groq_suggestions.get('mixtral', 'N/A')}\n"
            f"- Gemma Suggestion: {groq_suggestions.get('gemma', 'N/A')}"
        )

        final_output = await llm_gemini.ainvoke(merge_prompt.format(
            rag_section=rag_section_content, # Pass the constructed section
            additional_suggestions_section=additional_suggestions_section_content, # Pass this too
            **user_params # Continue passing these, as they are used in the main prompt body
        ))
    except Exception as e:
        logging.error(f"❌ Merge error: {e}", exc_info=True)
        final_output = "Something went wrong while combining AI suggestions. Please try again."

    # --- Session history and logging (remains same) ---
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
