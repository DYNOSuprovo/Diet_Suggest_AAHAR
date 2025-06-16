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
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from custom_callbacks import SafeTracer
from langchain_core.messages import AIMessage 
from typing import Optional 

# Google Sheets logging (Ensure gspread and oauth2client are installed: pip pip install gspread oauth2client)
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
    raise ValueError("GEMINI_API_KEY not set. Please set it in your .env file or environment.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

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
    version="0.2.4", # Incremented version
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Import Local Modules ---
from query_analysis import extract_all_metadata 

from llm_chains import (
    define_rag_prompt_template, setup_qa_chain, setup_conversational_qa_chain,
    define_merge_prompt_templates, get_session_history, define_generic_prompt
)
from embedding_utils import setup_vector_database
from groq_integration import cached_groq_answers

# --- LLM & Vector DB Setup ---
llm_gemini = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY
)

llm_generic = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY
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
    raise HTTPException(status_code=500, detail="Vector DB initialization failed. Check logs for details.")

# Setup LangChain components
rag_prompt = define_rag_prompt_template()
generic_prompt = define_generic_prompt() 

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
    # New optional fields for LLM parameters with validation
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Temperature for LLM creativity (0.0 to 1.0)")
    max_output_tokens: Optional[int] = Field(None, ge=1, le=8192, description="Max tokens in LLM response (e.g., 1 to 8192)")

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    user_query = chat_request.query
    client_session_id = chat_request.session_id
    
    # Get dynamic LLM parameters from request, use defaults if not provided
    llm_temperature = chat_request.temperature if chat_request.temperature is not None else 0.5 # Default temperature
    llm_max_output_tokens = chat_request.max_output_tokens if chat_request.max_output_tokens is not None else 2048 # Default max tokens

    # Generate or retrieve session ID for conversational memory
    session_id = client_session_id or request.session.get("session_id") or f"session_{os.urandom(8).hex()}"
    request.session["session_id"] = session_id 

    # Prepare config for LLM calls (must be defined AFTER session_id)
    llm_config = {
        "callbacks": [SafeTracer()], 
        "configurable": {
            "session_id": session_id,
        },
        "model_kwargs": { # Correct location for LLM parameters for GoogleGenerativeAI
            "temperature": llm_temperature,
            "max_output_tokens": llm_max_output_tokens
        }
    }

    logging.info(f"📩 Incoming Query: '{user_query}' | Session: {session_id} | Temp: {llm_temperature} | Max Tokens: {llm_max_output_tokens}")

    # Use the enhanced query_analysis to get all metadata
    query_metadata = extract_all_metadata(user_query)
    logging.info(f"🔍 Query Metadata: {query_metadata}")

    response_text = ""
    
    try:
        # --- Intent-Based Routing ---
        if query_metadata["primary_intent_type"] == "greeting":
            response_text = "Namaste! How can I assist you with a healthy Indian diet today?"
        
        elif query_metadata["primary_intent_type"] == "generic":
            logging.info("Handling generic query.")
            generic_response_obj = await llm_generic.ainvoke(
                generic_prompt.format(query=user_query),
                config=llm_config # Pass dynamic config
            )
            response_text = generic_response_obj.content if isinstance(generic_response_obj, AIMessage) else str(generic_response_obj)
            
        elif query_metadata["primary_intent_type"] == "formatting" and query_metadata["is_follow_up"]:
            chat_history = get_session_history(session_id).messages # Retrieve history inside the block
            last_ai_message_content = None
            for msg in reversed(chat_history):
                # Heuristic to find a substantial AI diet plan message
                if msg.type == 'ai' and msg.content and len(msg.content) > 50 and not msg.content.startswith("Namaste!") and not msg.content.startswith("I'm an AI"):
                    last_ai_message_content = msg.content
                    break

            if last_ai_message_content:
                logging.info("Reformatting previous response.")
                if query_metadata["wants_table"]:
                    merge_prompt_for_reformat = merge_prompt_table
                elif query_metadata["wants_paragraph"]:
                    merge_prompt_for_reformat = merge_prompt_paragraph
                else: 
                    merge_prompt_for_reformat = merge_prompt_default

                format_kwargs = {
                    "rag_section": f"Previous Answer:\n{last_ai_message_content}",
                    "additional_suggestions_section": "No new suggestions needed for reformatting.", # Empty for reformat
                    "query": user_query, 
                    "dietary_type": query_metadata.get("dietary_type", "any"), 
                    "goal": query_metadata.get("goal", "general"),     
                    "region": query_metadata.get("region", "Indian"),    
                    "disease_section": f"Disease Condition: {query_metadata['disease']}\n" if query_metadata.get("disease") else ""
                }
                
                reformatted_result = await llm_gemini.ainvoke(
                    merge_prompt_for_reformat.format(**format_kwargs),
                    config=llm_config # Pass dynamic config
                )
                response_text = reformatted_result.content if isinstance(reformatted_result, AIMessage) else str(reformatted_result)
            else:
                response_text = "I can only re-format a previous diet plan. Please ask for a diet plan first!"
                
            # Add user and AI messages to session history for formatting queries too
            get_session_history(session_id).add_user_message(user_query)
            get_session_history(session_id).add_ai_message(response_text)
            
            # Log to Google Sheet for formatting queries as well, then return
            try:
                if sheet_enabled and sheet:
                    sheet.append_row([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        session_id,
                        user_query,
                        response_text 
                    ])
                    logging.info("📝 Logged query and response to Google Sheet.")
            except Exception as e:
                logging.warning(f"⚠️ Google Sheet logging failed: {e}", exc_info=True)
            
            return JSONResponse(content={"answer": response_text, "session_id": session_id})


        else: # primary_intent_type is "task" or a formatting request with task keywords
            logging.info("Handling task-oriented query (RAG + Groq + Merge).")
            
            user_params = { 
                "dietary_type": query_metadata["dietary_type"],
                "goal": query_metadata["goal"],
                "region": query_metadata["region"],
                "disease": query_metadata["disease"] 
            }
            
            chat_history = get_session_history(session_id).messages # Retrieve history here for RAG

            rag_output_content = "No answer from RAG." 
            try:
                rag_result = await conversational_qa_chain.ainvoke({
                    "query": user_query,
                    "chat_history": chat_history, # Use chat_history here
                    **user_params 
                }, config=llm_config)
                
                rag_output_content = rag_result.get("answer", "No answer from RAG chain.") 
                logging.info(f"✅ RAG Chain Raw Output (from 'answer' key): {rag_output_content[: min(len(rag_output_content), 500)]}...") 
            except Exception as e:
                logging.error(f"❌ RAG error during ainvoke: {type(e).__name__}: {e}", exc_info=True)
                rag_output_content = "Error while retrieving response from knowledge base."

            # Define common phrases where RAG indicates it needs more info
            # Keep these specific to RAG's refusal/context-lacking messages
            rag_refusal_phrases = [
                "I need more information",
                "please provide more details",
                "context insufficient",
                "cannot provide a detailed plan",
                "please specify",
                "not suitable for everyone", 
                "only indicates a general request" 
            ]
            
            # Convert to lowercase for case-insensitive check
            rag_output_lower = rag_output_content.lower()

            # Check if RAG chain explicitly requested more info, if so, handle it here directly
            if any(phrase in rag_output_lower for phrase in rag_refusal_phrases):
                logging.info(f"💡 RAG chain indicated more information is needed. Skipping Groq/Merge and returning specific guidance.")
                response_text = "To provide a tailored diet plan, I need a bit more detail. Could you please specify things like age, gender, activity level, or any dietary preferences (vegetarian, non-vegetarian, vegan, etc.) or health conditions (like diabetes, high BP)? This will help me give you a more personalized and effective plan."
                
                # Add user and AI messages to session history for task-refusal queries
                get_session_history(session_id).add_user_message(user_query)
                get_session_history(session_id).add_ai_message(response_text)
                
                # Log to Google Sheet for task-refusal queries, then return
                try:
                    if sheet_enabled and sheet:
                        sheet.append_row([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            session_id,
                            user_query,
                            response_text 
                        ])
                        logging.info("📝 Logged query and response to Google Sheet.")
                except Exception as e:
                    logging.warning(f"⚠️ Google Sheet logging failed: {e}", exc_info=True)
                
                return JSONResponse(content={"answer": response_text, "session_id": session_id})
            else:
                # Proceed with Groq and Merge only if RAG provided a substantive answer
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

                if query_metadata["wants_table"]:
                    merge_prompt = merge_prompt_table
                elif query_metadata["wants_paragraph"]:
                    merge_prompt = merge_prompt_paragraph
                else:
                    merge_prompt = merge_prompt_default

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
                        "disease_section": f"Disease Condition: {user_params['disease']}\n" if user_params.get("disease") else ""
                    }
                    
                    formatted_prompt_string = merge_prompt.format(**format_kwargs)
                    logging.info(f"➡️ Sending Merge Prompt (length: {len(formatted_prompt_string)} chars): {formatted_prompt_string[: min(len(formatted_prompt_string), 1000)]}...") 

                    merge_result_obj = await llm_gemini.ainvoke(
                        formatted_prompt_string, 
                        config=llm_config 
                    )
                    response_text = merge_result_obj.content if isinstance(merge_result_obj, AIMessage) else str(merge_result_obj)
                except Exception as e:
                    logging.error(f"❌ Merge process error: {type(e).__name__}: {e}", exc_info=True) 
                    final_output_content = "I encountered an issue generating a comprehensive diet plan. Please try again."
                
                response_text = final_output_content

    except Exception as e:
        logging.error(f"❌ Unhandled error in /chat endpoint: {e}", exc_info=True)
        response_text = "I'm sorry, an unexpected error occurred. Please try again."

    # The following block is now redundant and should be removed or moved inside specific branches
    # get_session_history(session_id).add_user_message(user_query)
    # get_session_history(session_id).add_ai_message(response_text)

    # try:
    #     if sheet_enabled and sheet:
    #         sheet.append_row([
    #             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #             session_id,
    #             user_query,
    #             response_text 
    #         ])
    #         logging.info("📝 Logged query and response to Google Sheet.")
    # except Exception as e:
    #     logging.warning(f"⚠️ Google Sheet logging failed: {e}", exc_info=True)

    # return JSONResponse(content={"answer": response_text, "session_id": session_id})

    # The final logging and return must happen consistently for all paths
    logging.info(f"Final response for session {session_id}: {response_text[:200]}...") # Log final response
    try:
        get_session_history(session_id).add_user_message(user_query)
        get_session_history(session_id).add_ai_message(response_text)
    except Exception as e:
        logging.error(f"❌ Error adding to chat history: {e}", exc_info=True)
        # This error shouldn't stop the response, just log it.

    try:
        if sheet_enabled and sheet:
            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_id,
                user_query,
                response_text 
            ])
            logging.info("📝 Logged query and response to Google Sheet.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheet logging failed: {e}", exc_info=True)

    return JSONResponse(content={"answer": response_text, "session_id": session_id})
