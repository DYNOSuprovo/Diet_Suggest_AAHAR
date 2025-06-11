import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.extractors import extract_diet_goal, extract_diet_preference, extract_regional_preference, contains_table_request
from utils.session_memory import get_session_history
from utils.google_logger import sheet
from llms.groq_suggestions import cached_groq_answers
from llms.merge_final import merge_prompt_default, merge_prompt_table, llm_gemini
from rag.init_chain import conversational_qa_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class ChatRequest(BaseModel):
    query: str
    session_id: str = None


@app.get("/")
async def health_check():
    return {"status": "ok"}


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

    try:
        rag_result = await conversational_qa_chain.ainvoke({
            "query": user_query,
            "chat_history": chat_history
        }, config={"configurable": {"session_id": session_id}})
        rag_output = rag_result.get("answer", "No answer from RAG.")
    except Exception as e:
        logging.error(f"❌ RAG error: {e}", exc_info=True)
        rag_output = "Error while retrieving response from knowledge base."

    groq_suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
    if GROQ_API_KEY:
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
            groq_suggestions = {k: f"Error: {e}" for k in groq_suggestions}

    try:
        merge_prompt = merge_prompt_table if contains_table_request(user_query) else merge_prompt_default
        merge_input = {
            "rag": rag_output,
            "llama": groq_suggestions.get("llama", "N/A"),
            "mixtral": groq_suggestions.get("mixtral", "N/A"),
            "gemma": groq_suggestions.get("gemma", "N/A"),
            **user_params
        }
        final_output = await llm_gemini.ainvoke(merge_prompt.format(**merge_input))
    except Exception as e:
        logging.error(f"❌ Merge error: {e}", exc_info=True)
        final_output = "Something went wrong while combining AI suggestions."

    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(final_output)

    try:
        if sheet:
            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_id,
                user_query,
                final_output
            ])
            logging.info("📝 Logged to Google Sheet.")
    except Exception as log_err:
        logging.error(f"❌ Failed to log to sheet: {log_err}")

    return JSONResponse(content={"answer": final_output, "session_id": session_id})
