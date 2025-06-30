# Please paste your FastAPI code here. I'm ready to help you fix any errors!
# fastapi_app.py - Advanced Agentic Version for Single-File Deployment

import os
import json
import logging
import zipfile
import requests
import string
import re
import base64
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Langchain and Google Generative AI imports
# Updated Chroma import to address deprecation warning
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # Added RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import StringPromptValue # Import StringPromptValue


# Google Sheets logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Consolidated: Custom Callback for LangChain ---
class SafeTracer(BaseCallbackHandler):
    """
    A custom LangChain callback handler to safely log chain outputs.
    Avoids breaking if output structure is unexpected.
    Logs to Python's logging system.
    """
    def on_chain_end(self, outputs: Any, **kwargs): # Changed outputs type hint to Any for broader compatibility
        try:
            if isinstance(outputs, (AIMessage, HumanMessage, SystemMessage)):
                logging.info(f"🔁 Chain ended. Message type: {type(outputs).__name__}, content snippet: {outputs.content[:100]}...")
            elif isinstance(outputs, StringPromptValue): # Handle StringPromptValue specifically
                logging.info(f"🔁 Chain ended. PromptValue content snippet: {outputs.text[:100]}...")
            elif isinstance(outputs, dict):
                if "answer" in outputs:
                    logging.info(f"🔁 Chain ended. Answer snippet: {outputs['answer'][:100]}...")
                elif "output" in outputs:
                    logging.info(f"🔁 Chain ended. Output snippet: {outputs['output'][:100]}...")
                elif "text" in outputs: # Common for simple string outputs from LLMs within a dict
                    logging.info(f"🔁 Chain ended. Text output snippet: {outputs['text'][:100]}...")
                else: # Fallback for any other dictionary structure
                    logging.info(f"🔁 Chain ended. Output (type: {type(outputs)}, content snippet): {str(outputs)[:100]}...")
            elif isinstance(outputs, str): # Direct string output
                logging.info(f"🔁 Chain ended. String output snippet: {outputs[:100]}...")
            else:
                # Catch-all for any other unexpected types
                logging.info(f"🔁 Chain ended. Output (type: {type(outputs)}, content snippet): {str(outputs)[:100]}...")
        except Exception as e:
            logging.error(f"❌ Error in on_chain_end callback: {e}")


# --- Consolidated: Vector Database Setup & Download ---
def download_and_extract_db_for_app():
    """
    Downloads and extracts a prebuilt ChromaDB from a HuggingFace URL.
    This function checks if the DB already exists to avoid re-downloading on restarts.
    """
    url = "https://huggingface.co/datasets/Dyno1307/chromadb-diet/resolve/main/db.zip"
    zip_path = "/tmp/db.zip" # Using /tmp for temporary storage on Render
    extract_path = "/tmp/chroma_db" # MUST match setup_vector_database's persist_directory

    os.makedirs(extract_path, exist_ok=True)

    if os.path.exists(os.path.join(extract_path, "index")): # Check for 'index' file within the extracted path
        logging.info("✅ Chroma DB already exists, skipping download.")
        return

    try:
        logging.info("⬇️ Downloading Chroma DB zip from HuggingFace...")
        with requests.get(url, stream=True, timeout=120) as r: # Increased timeout for large files
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info(f"📦 Extracting zip to {extract_path}...")
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

def setup_vector_database(chroma_db_directory: str = "/tmp/chroma_db", in_memory: bool = False):
    """
    Initializes Chroma vector database using Gemini embeddings.
    """
    try:
        logging.info("🔧 Initializing Gemini Embeddings...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment variables. Please set it for Gemini API access.")
        
        # FIX: Corrected typo in GoogleGenerativeAIEmbeddings
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        logging.info("✅ Gemini Embeddings loaded.")

        persist_path = None if in_memory else chroma_db_directory

        db = Chroma(
            persist_directory=persist_path,
            embedding_function=embedding
        )

        try:
            count = len(db.get()['documents'])
            logging.info(f"📦 Vector DB loaded with {count} documents.")
        except Exception as e:
            logging.warning(f"⚠️ Could not count documents in Vector DB (might be empty or first run): {e}")

        logging.info("✅ Chroma DB initialized successfully.")
        return db, embedding

    except Exception as e:
        logging.exception("❌ Vector DB setup failed.")
        raise

# --- Consolidated: Groq Integration ---
def cached_groq_answers(query: str, groq_api_key: str, dietary_type: str, goal: str, region: str) -> dict:
    """
    Fetches diet suggestions from multiple Groq models in parallel.
    Uses ThreadPoolExecutor for concurrent synchronous API calls.
    """
    logging.info(f"Fetching Groq answers for query: '{query}', pref: '{dietary_type}', goal: '{goal}', region: '{region}'")
    # Removed 'mixtral' due to persistent 404 errors, kept 'mistral-saba' as requested and validated.
    models = ["llama", "gemma", "mistral-saba"] 
    results = {}
    if not groq_api_key:
        logging.warning("GROQ_API_KEY not available. Skipping Groq calls.")
        return {k: "Groq API key not available." for k in models}

    def _groq_diet_answer_single(model_name: str):
        """Helper function to call a single Groq model."""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
            groq_model_map = {
                "llama": "llama3-70b-8192",
                "gemma": "gemma2-9b-it",
                "mistral-saba": "mistral-saba-24b" # Confirmed this model name is correct and available
            }
            actual_model_name = groq_model_map.get(model_name.lower(), model_name)
            
            prompt_content = (
                f"User query: '{query}'. "
                f"Provide a concise, practical {dietary_type} diet suggestion or food item "
                f"for {goal}, tailored for a {region} Indian context. "
                f"Focus on readily available ingredients. Be brief and to the point."
            )
            payload = {
                "model": actual_model_name,
                "messages": [{"role": "user", "content": prompt_content}],
                "temperature": 0.5,
                "max_tokens": 250
            }
            logging.info(f"Calling Groq API: {actual_model_name} for query: '{query}' ({dietary_type} {goal}, {region})")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data and data.get('choices') and data['choices'][0].get('message'):
                return data['choices'][0]['message']['content']
            
            return f"No suggestion from {actual_model_name} (empty/malformed response)."
        
        except requests.exceptions.Timeout:
            logging.error(f"Timeout error from {model_name} for query: '{query}'")
            return f"Timeout error from {model_name}."
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error from {model_name} for query: '{query}': {e}")
            return f"Request error from {model_name}: {e}"
        except Exception as e:
            logging.error(f"Unexpected error from {model_name} for query: '{query}': {e}")
            return f"Error from {model_name}: {e}"

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {executor.submit(_groq_diet_answer_single, name): name for name in models}
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
            except Exception as e:
                logging.error(f"ThreadPool error for {model_name}: {e}")
                results[model_name] = f"Failed to get result: {e}"
    return results

# --- Consolidated: LangChain Chain Definitions ---
llm_chains_session_store = {} 

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or creates a LangChain ChatMessageHistory for a given session ID."""
    if session_id not in llm_chains_session_store:
        logging.info(f"Creating new Langchain session history in 'llm_chains_session_store' for: {session_id}")
        llm_chains_session_store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history from 'llm_chains_session_store' for: {session_id}")
    return llm_chains_session_store[session_id]

def define_rag_prompt_template():
    """Defines the prompt template for the RAG chain."""
    template_string = """
    You are an AI assistant specialized in Indian diet and nutrition created by Suprovo.
    Based on the following conversation history and the user's query, provide a simple, practical, and culturally relevant **{dietary_type}** food suggestion suitable for Indian users aiming for **{goal}**.
    If a specific region like **{region}** is mentioned or inferred, prioritize food suggestions from that region.
    Focus on readily available ingredients and common Indian dietary patterns for the specified region.
    Be helpful, encouraging, and specific where possible.
    Use the chat history to understand the context of the user's current query and maintain continuity.
    Strictly adhere to the **{dietary_type}** and **{goal}** requirements, and the **{region}** preference if specified.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    User Query:
    {query}

    {dietary_type} {goal} Food Suggestion (Tailored for {region} Indian context):
    """
    return PromptTemplate(
        template=template_string,
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "context"]
    )

def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    """Sets up the Retrieval Augmented Generation (RAG) chain."""
    try:
        retriever = db.as_retriever(search_kwargs={"k": 5})

        def retrieve_and_log_context(input_dict):
            """Helper to retrieve documents and log their content."""
            docs = retriever.invoke(input_dict["query"])
            if not docs:
                logging.warning(f"No documents retrieved for query: '{input_dict['query']}'")
            context_str = "\n\n".join(doc.page_content for doc in docs)
            logging.info(f"Retrieved Context (snippet): {context_str[:200]}...")
            return context_str

        qa_chain = (
            {
                "context": retrieve_and_log_context,
                "query": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "dietary_type": RunnablePassthrough(),
                "goal": RunnablePassthrough(),
                "region": RunnablePassthrough(),
            }
            | rag_prompt
            | llm_gemini
            | StrOutputParser()
        )
        logging.info("Retrieval QA Chain initialized successfully.")
        return qa_chain
    except Exception as e:
        logging.exception("Full QA Chain setup traceback:")
        raise RuntimeError(f"QA Chain setup error: {e}")

def setup_conversational_qa_chain(qa_chain):
    """Wraps the QA chain with message history capabilities."""
    conversational_qa_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    logging.info("Conversational QA Chain initialized.")
    return conversational_qa_chain

def define_merge_prompt_templates():
    """Defines prompt templates for merging RAG and Groq outputs."""
    merge_prompt_default_template = """
    You are an AI assistant specialized in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
    3. Combine information logically and seamlessly, without mentioning the source of each piece.
    4. Ensure the final plan is clear, actionable, and culturally relevant.
    5. If the user's input was only a greeting, respond politely without providing a diet plan.

    Final {dietary_type} {goal} Food Suggestion/Diet Plan (Tailored for {region} Indian context):
    """
    
    merge_prompt_table_template = """
    You are an AI assistant specialized in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.
    **You MUST present the final diet plan as a clear markdown table. Include columns for Meal, Food Items, and Notes/Considerations.**

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
    3. Combine information logically and seamlessly, without mentioning the source of each piece.
    4. Ensure the final plan is clear, actionable, and culturally relevant.
    5. If the user's input was only a greeting, respond politely without providing a diet plan.

    Final {dietary_type} {goal} Diet Plan (Tailored for {region} Indian context, in markdown table format):
    """

    logging.info("Merge Prompt templates created.")
    return (
        PromptTemplate(template=merge_prompt_default_template, input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"]),
        PromptTemplate(template=merge_prompt_table_template, input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"])
    )

# --- Consolidated: Query Analysis and Intent Detection (now primarily for sub-tool use) ---
def clean_query(query: str) -> str:
    """Strips punctuation and lowercases the query for consistent keyword matching."""
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

@lru_cache(maxsize=128)
def extract_diet_preference(query: str) -> str:
    """Extracts dietary preference (vegan, vegetarian, non-vegetarian) from the query."""
    q = query.lower()
    if any(x in q for x in ["non-veg", "non veg", "nonvegetarian"]):
        return "non-vegetarian"
    if "vegan" in q:
        return "vegan"
    if "veg" in q or "vegetarian" in q:
        return "vegetarian"
    return "any"

@lru_cache(maxsize=128)
def extract_diet_goal(query: str) -> str:
    """Extracts diet goal (weight loss, weight gain, general diet) from the query."""
    q = query.lower()
    if any(p in q for p in ["lose weight", "loss weight", "cut weight", "reduce weight", "lose fat", "cut fat"]):
        return "weight loss"
    if "gain weight" in q or "weight gain" in q or "muscle gain" in q:
        return "weight gain"
    if "loss" in q:
        return "weight loss"
    if "gain" in q:
        return "weight gain"
    return "diet"

@lru_cache(maxsize=128)
def extract_regional_preference(query: str) -> str:
    """Extracts regional preference for Indian diet (e.g., South Indian, Punjabi)."""
    q = query.lower()
    match = re.search(
        r"\b(?:south indian|north indian|west indian|east indian|bengali|punjabi|maharashtrian|gujarati|"
        r"tamil|kannada|telugu|malayalam|kanyakumari|odisha|oriya|bhubaneswar|cuttack|angul)\b",
        q
    )
    if match:
        return " ".join(word.capitalize() for word in match.group(0).split())
    return "Indian"

@lru_cache(maxsize=128)
def contains_table_request(query: str) -> bool:
    """Checks if the query explicitly asks for a tabular format."""
    q = query.lower()
    return any(k in q for k in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])

def detect_sentiment(llm_instance: GoogleGenerativeAI, query: str) -> str:
    """Detects the sentiment of the user's query."""
    prompt = f"""
    Analyze the sentiment of the following user query. Respond with only one word: 'positive', 'neutral', or 'negative'.

    Query: "{query}"

    Sentiment:
    """
    try:
        response_obj = llm_instance.invoke(prompt)
        
        # FIX: Handle cases where llm.invoke directly returns a string instead of AIMessage
        if isinstance(response_obj, AIMessage):
            sentiment = response_obj.content.strip().lower()
        elif isinstance(response_obj, str):
            sentiment = response_obj.strip().lower()
        else:
            logging.warning(f"LLM returned unexpected type for sentiment: {type(response_obj)} for query '{query}'. Defaulting to 'neutral'.")
            return "neutral"

        if sentiment in ["positive", "neutral", "negative"]:
            return sentiment
        else:
            logging.warning(f"LLM returned unexpected sentiment: '{sentiment}' for query '{query}'. Defaulting to 'neutral'.")
            return "neutral"
    except Exception as e:
        logging.error(f"Error detecting sentiment for query '{query}': {e}", exc_info=True)
        return "neutral"

# --- New: Placeholder Tools for Agent ---
async def tool_fetch_recipe(recipe_name: str) -> str:
    """
    Placeholder tool to simulate fetching a recipe.
    In a real application, this would query a recipe API or database.
    """
    logging.info(f"Executing tool: fetch_recipe for '{recipe_name}'")
    # Simulate API call delay
    import asyncio
    await asyncio.sleep(0.5) 
    if "dal makhani" in recipe_name.lower():
        return f"Recipe for {recipe_name}: Ingredients - Black lentils, kidney beans, butter, cream, tomatoes, ginger-garlic paste. Steps - Soak, boil, temper, simmer. Serve hot with naan or rice."
    elif "paneer tikka" in recipe_name.lower():
        return f"Recipe for {recipe_name}: Ingredients - Paneer, yogurt, ginger-garlic paste, spices, bell peppers, onions. Steps - Marinate, skewer, grill/bake."
    else:
        return f"Recipe for {recipe_name}: Detailed recipe unavailable, but typically involves [basic ingredients] and [basic cooking method]."

async def tool_lookup_nutrition_facts(food_item: str) -> str:
    """
    Placeholder tool to simulate looking up nutrition facts.
    In a real application, this would query a nutrition API or database.
    """
    logging.info(f"Executing tool: lookup_nutrition_facts for '{food_item}'")
    import asyncio
    await asyncio.sleep(0.5)
    if "rice" in food_item.lower():
        return f"Nutrition facts for {food_item} (per 100g cooked): Calories: 130, Carbs: 28g, Protein: 2.7g, Fat: 0.3g."
    elif "lentils" in food_item.lower():
        return f"Nutrition facts for {food_item} (per 100g cooked): Calories: 116, Carbs: 20g, Protein: 9g, Fat: 0.4g. Rich in fiber."
    else:
        return f"Nutrition facts for {food_item}: Calories, carbs, protein, and fat vary. Generally healthy."

# --- Agentic Orchestration Pydantic Model ---
class AgentAction(BaseModel):
    """
    Represents the action the AI Agent decides to take or the final answer it provides.
    """
    thought: str = Field(..., description="A brief thought process explaining the current decision.")
    tool_name: Optional[str] = Field(None, description="The name of the tool to use. Must be one of: 'generate_diet_plan', 'reformat_diet_plan', 'handle_greeting', 'handle_identity', 'fetch_recipe', 'lookup_nutrition_facts'.")
    tool_input: Optional[Dict[str, Any]] = Field(None, description="A dictionary of parameters for the selected tool.")
    final_answer: Optional[str] = Field(None, description="The final answer to the user's request. Only set this if the task is complete.")

# --- Agentic Orchestrator Prompt ---
ORCHESTRATOR_PROMPT_TEMPLATE = """
You are an intelligent AI agent named AAHAR, specialized in Indian diet and nutrition.
Your goal is to assist users with their diet-related queries by thinking step-by-step, deciding on appropriate actions, and providing comprehensive answers.
You can use various tools to gather information or perform tasks.

You must always respond with a JSON object that adheres to the `AgentAction` Pydantic model.
Your response MUST include a `thought` describing your reasoning.
You must either provide a `final_answer` OR specify a `tool_name` and `tool_input`.
DO NOT use both `tool_name` and `final_answer` in the same response.

Available Tools:
1.  **`handle_greeting`**:
    * **Description**: Respond to simple greetings (e.g., "Hi", "Hello", "Namaste").
    * **Input**: None needed.
    * **When to use**: If the user's query is purely a greeting.
2.  **`handle_identity`**:
    * **Description**: Respond to queries asking about your identity or creator (e.g., "Who are you?", "Who made you?").
    * **Input**: None needed.
    * **When to use**: If the user's query is about your identity.
3.  **`reformat_diet_plan`**:
    * **Description**: Reformat a *previous* diet plan provided by the AI (e.g., "Can you put that in a table?", "List it out for me").
    * **Input**: `wants_table: boolean` (true if user specifically asked for a table format).
    * **When to use**: Only if there is a substantial previous AI message in the `chat_history` that looks like a diet plan, AND the user is explicitly asking to reformat it.
4.  **`generate_diet_plan`**:
    * **Description**: Generate a new diet suggestion or detailed diet plan using RAG and Groq.
    * **Input**: `dietary_type: string` (e.g., "vegetarian", "non-vegetarian", "vegan", "any"), `goal: string` (e.g., "weight loss", "weight gain", "diet"), `region: string` (e.g., "South Indian", "Punjabi", "Indian"), `wants_table: boolean`. Default to "any", "diet", "Indian", false if not specified by user.
    * **When to use**: For most diet-related queries that require generating a new plan or suggestion.
5.  **`fetch_recipe`**:
    * **Description**: Fetch a simple recipe for a given food item.
    * **Input**: `recipe_name: string` (e.g., "Dal Makhani").
    * **When to use**: If the user asks for a specific recipe.
6.  **`lookup_nutrition_facts`**:
    * **Description**: Look up basic nutrition facts for a given food item.
    * **Input**: `food_item: string` (e.g., "rice", "lentils").
    * **When to use**: If the user asks for nutritional information about a food.

**Agent's State:**
You have access to the current `chat_history` and `current_user_query`.
You also have an `agent_scratchpad` which contains past `Tool Output` to help you make subsequent decisions.

Chat History:
{chat_history}

Current User Query: "{query}"

Agent Scratchpad (Observations from previous tool executions):
{agent_scratchpad}

Think step-by-step. What is the user's ultimate goal? What is the next logical step to achieve that goal?
If you've gathered all necessary information and are ready to answer, set `final_answer`.
Otherwise, select the `tool_name` and `tool_input` for your next action.

Output (JSON adhering to AgentAction Pydantic model):
"""

ORCHESTRATOR_PROMPT = PromptTemplate(
    template=ORCHESTRATOR_PROMPT_TEMPLATE,
    input_variables=["chat_history", "query", "agent_scratchpad"],
    partial_variables={"format_instructions": JsonOutputParser(pydantic_object=AgentAction).get_format_instructions()}
)

# --- Main FastAPI Application Setup ---

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_secure_random_key_CHANGE_THIS_IN_PRODUCTION")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Diet Suggest Logs")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
# Suppress the new langchain_chroma warnings if desired (will still appear from old langchain_community if both present)
logging.getLogger('langchain_chroma.base').setLevel(logging.WARNING)


# --- Google Sheets Setup ---
sheet = None
sheet_enabled = False
try:
    if GOOGLE_CREDS_JSON:
        # Strip whitespace before decoding, as padding errors can be caused by extra chars
        # CRITICAL: Ensure GOOGLE_CREDS_JSON on Render is a perfectly base64-encoded string
        # with no leading/trailing spaces, newlines, or other unintended characters.
        creds_dict = json.loads(base64.b64decode(GOOGLE_CREDS_JSON.strip()).decode('utf-8')) 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive",
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/spreadsheets"]
        )
        gs_client = gspread.authorize(creds)
        sheet = gs_client.open(GOOGLE_SHEET_NAME).sheet1
        sheet_enabled = True
        logging.info("✅ Google Sheets connected for logging.")
    else:
        logging.warning("⚠️ GOOGLE_CREDS_JSON environment variable not set. Google Sheets disabled.")
except (json.JSONDecodeError, UnicodeDecodeError, base64.binascii.Error) as e: # Added base64 error
    logging.warning(f"⚠️ Error decoding or parsing GOOGLE_CREDS_JSON: {e}. Google Sheets disabled. Please check your base64 encoding for this environment variable.")
except Exception as e:
    logging.warning(f"⚠️ Google Sheets connection failed: {e}. Logging to sheet disabled.", exc_info=True)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions using RAG and LLMs.",
    version="0.3.0", # Increment version for advanced agentic routing
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Initialize LLM & Vector DB at Application Startup ---
llm_gemini: Optional[GoogleGenerativeAI] = None
llm_orchestrator: Optional[GoogleGenerativeAI] = None
db: Optional[Chroma] = None
rag_prompt: Optional[PromptTemplate] = None
qa_chain: Optional[Any] = None # Using Any for Runnable type
conversational_qa_chain: Optional[Any] = None
merge_prompt_default: Optional[PromptTemplate] = None
merge_prompt_table: Optional[PromptTemplate] = None
orchestrator_chain: Optional[Any] = None

@app.on_event("startup")
async def startup_event():
    """
    Function to run during FastAPI application startup.
    Initializes LLMs, downloads/sets up Vector DB, and configures LangChain components.
    """
    global llm_gemini, llm_orchestrator, db, rag_prompt, qa_chain, conversational_qa_chain, \
           merge_prompt_default, merge_prompt_table, orchestrator_chain

    try:
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is not set. Please provide it in your environment variables on Render.")
        
        llm_gemini = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.5
        )
        llm_orchestrator = GoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=GEMINI_API_KEY,
            temperature=0.1 # Very low temperature for consistent JSON output from orchestrator
        )
        logging.info("✅ Gemini LLMs initialized.")
    except Exception as e:
        logging.error(f"❌ Gemini LLM initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini LLM initialization failed.")

    try:
        download_and_extract_db_for_app()
    except HTTPException as e:
        logging.error(f"❌ Fatal error during DB download/extraction: {e.detail}")
        raise

    try:
        db, _ = setup_vector_database(chroma_db_directory="/tmp/chroma_db")
        logging.info(f"✅ Vector DB initialized from '/tmp/chroma_db'.")
    except Exception as e:
        logging.error(f"❌ Vector DB init error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Vector DB initialization failed.")

    try:
        rag_prompt = define_rag_prompt_template()
        qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
        conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
        merge_prompt_default, merge_prompt_table = define_merge_prompt_templates()

        def parse_agent_action_output(llm_output: Union[AIMessage, str]) -> AgentAction:
            """
            Parses the LLM output (which should be JSON) into an AgentAction Pydantic model.
            Handles cases where LLM output might contain markdown fences or extra text.
            """
            parser = JsonOutputParser(pydantic_object=AgentAction)
            content_str = llm_output.content if isinstance(llm_output, AIMessage) else str(llm_output)
            
            # Attempt to extract JSON from markdown code block if present
            json_match = re.search(r"```json\n(.*)\n```", content_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = content_str.strip() # Assume it's direct JSON

            try:
                parsed_output = parser.parse(json_str)
                # Ensure it's an AgentAction instance, not just a dict
                if isinstance(parsed_output, dict):
                    return AgentAction(**parsed_output)
                elif isinstance(parsed_output, AgentAction):
                    return parsed_output
                else:
                    logging.error(f"Parsed output is neither a dict nor AgentAction instance. Type: {type(parsed_output)}, Content: {parsed_output}")
                    return AgentAction(
                        thought="Orchestrator returned unexpected structure, defaulting to error state.",
                        tool_name=None,
                        tool_input=None,
                        final_answer="An internal system error occurred due to unexpected output from the AI. Please try again."
                    )
            except ValidationError as e:
                logging.error(f"Pydantic validation error when parsing LLM output to AgentAction: {e}. Raw output: {json_str}")
                return AgentAction(
                    thought="Orchestrator returned invalid JSON format. Recalculating.",
                    tool_name=None,
                    tool_input=None,
                    final_answer="An internal system error occurred while processing your request due to invalid AI output. Please try again."
                )
            except Exception as e:
                logging.error(f"General error parsing LLM output to AgentAction: {e}. Raw output: {json_str}")
                return AgentAction(
                    thought="Orchestrator returned malformed JSON, defaulting to error state.",
                    tool_name=None,
                    tool_input=None,
                    final_answer="An internal system error occurred while processing your request. Please try again."
                )

        orchestrator_chain = (
            ORCHESTRATOR_PROMPT
            | llm_orchestrator
            | RunnableLambda(parse_agent_action_output) # Use RunnableLambda for robust parsing
        )
        logging.info("✅ LangChain QA Chains, Merge Prompts, and Orchestrator Chain initialized successfully.")
    except Exception as e:
        logging.error(f"❌ LangChain component setup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="LangChain component initialization failed.")

# --- Pydantic Schema for Request Body ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(None, description="Optional session ID from client. If not provided, a new one is generated.")

# --- API Endpoints ---

@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    """
    Main chat endpoint, implementing an LLM-driven agent loop.
    The agent iteratively decides on actions (tool use) or provides a final answer.
    """
    user_query = chat_request.query
    client_session_id = chat_request.session_id
    
    session_id = client_session_id or request.session.get("session_id") or f"session_{os.urandom(8).hex()}"
    request.session["session_id"] = session_id

    logging.info(f"📩 Query: '{user_query}' | Session: {session_id}")

    response_text = "I'm sorry, I encountered an internal issue and cannot respond right now. Please try again."
    sentiment = "neutral" # Default sentiment, updated during execution

    # Get current chat history for the session (LangChain Messages objects)
    chat_history_lc = get_session_history(session_id).messages

    # Convert chat_history_lc to a more readable string for the orchestrator prompt
    formatted_chat_history = ""
    for msg in chat_history_lc:
        if isinstance(msg, HumanMessage):
            formatted_chat_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_chat_history += f"AI: {msg.content}\n"

    # --- AGENT LOOP ---
    max_agent_iterations = 5 # Limit the number of steps to prevent infinite loops
    agent_scratchpad: List[Dict[str, Any]] = [] # Stores (tool_name, tool_input, tool_output) history

    try:
        for i in range(max_agent_iterations):
            logging.info(f"🔄 Agent Iteration {i+1}/{max_agent_iterations}")
            
            # Prepare scratchpad for prompt
            scratchpad_str = "\n".join([
                f"Tool Used: {item.get('tool_name')}\nTool Input: {item.get('tool_input')}\nTool Output: {item.get('tool_output')}"
                for item in agent_scratchpad
            ])

            # --- AGENTIC ROUTING: Invoke the Orchestrator LLM ---
            orchestrator_decision: AgentAction = await orchestrator_chain.ainvoke({
                "query": user_query,
                "chat_history": formatted_chat_history,
                "agent_scratchpad": scratchpad_str
            }, config={
                "callbacks": [SafeTracer()],
                "configurable": {"session_id": session_id}
            })
            
            logging.info(f"✨ Orchestrator Decision (Iter {i+1}): Thought='{orchestrator_decision.thought}' Tool='{orchestrator_decision.tool_name}', Params={orchestrator_decision.tool_input}")

            # If the orchestrator provides a final answer, break the loop
            if orchestrator_decision.final_answer:
                response_text = orchestrator_decision.final_answer
                logging.info(f"✅ Agent provided final answer on iteration {i+1}.")
                break # Exit the agent loop
            
            # Otherwise, execute the chosen tool
            tool_name = orchestrator_decision.tool_name
            tool_input = orchestrator_decision.tool_input if orchestrator_decision.tool_input is not None else {}
            tool_output = "Error: Tool execution failed." # Default if tool fails

            try: 
                if tool_name == "handle_greeting":
                    response_text = "Namaste! How can I assist you with a healthy Indian diet today?"
                    tool_output = response_text
                    break # Task complete (single-turn greeting)
                elif tool_name == "handle_identity":
                    response_text = "I am an AI assistant specialized in Indian diet and nutrition, created by Suprovo."
                    tool_output = response_text
                    break # Task complete (single-turn identity)
                elif tool_name == "reformat_diet_plan":
                    logging.info("Executing tool: reformat_diet_plan.")
                    last_ai_message_content = None
                    for msg in reversed(chat_history_lc):
                        if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 50:
                            last_ai_message_content = msg.content
                            break

                    if last_ai_message_content:
                        wants_table_flag = tool_input.get("wants_table", False)
                        merge_prompt_template = merge_prompt_table if wants_table_flag else merge_prompt_default
                        reformat_response_obj = await llm_gemini.ainvoke(
                            merge_prompt_template.format(
                                rag_section=f"Previous Answer to Reformat:\n{last_ai_message_content}",
                                additional_suggestions_section="No new suggestions needed for reformatting, just reformat the above.",
                                dietary_type=tool_input.get("dietary_type", "any"),
                                goal=tool_input.get("goal", "diet"),     
                                region=tool_input.get("region", "Indian"),    
                            ),
                            config={
                                "callbacks": [SafeTracer()],
                                "configurable": {"session_id": session_id}
                            }
                        )
                        tool_output = reformat_response_obj.content if isinstance(reformat_response_obj, AIMessage) else str(reformat_response_obj)
                        response_text = tool_output # Reformatting provides the final answer for that request
                        break 
                    else:
                        tool_output = "I cannot reformat. No substantial previous AI message found in the chat history that looks like a diet plan."
                        response_text = tool_output 
                        break

                elif tool_name == "generate_diet_plan":
                    logging.info("Executing tool: generate_diet_plan (RAG + Groq + Merge pipeline).")
                    user_params = {
                        "dietary_type": tool_input.get("dietary_type", "any"),
                        "goal": tool_input.get("goal", "diet"),
                        "region": tool_input.get("region", "Indian"),
                    }
                    wants_table_flag = tool_input.get("wants_table", False)

                    rag_output_content = "Error from RAG."
                    try:
                        rag_result = await conversational_qa_chain.ainvoke({
                            "query": user_query,
                            "chat_history": chat_history_lc,
                            **user_params
                        }, config={
                            "callbacks": [SafeTracer()],
                            "configurable": {"session_id": session_id}
                        })
                        rag_output_content = str(rag_result)
                    except Exception as e:
                        logging.error(f"❌ RAG error in tool: {e}", exc_info=True)
                        rag_output_content = "Error while retrieving response from knowledge base."

                    groq_suggestions = {}
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
                            logging.error(f"❌ Groq error in tool: {e}", exc_info=True)
                            groq_suggestions = {"llama": "Error", "gemma": "Error", "mistral-saba": "Error"} 

                    merge_prompt_template = merge_prompt_table if wants_table_flag else merge_prompt_default
                    try:
                        merge_input_kwargs = {
                            "rag_section": f"Primary RAG Answer:\n{rag_output_content}",
                            "additional_suggestions_section": (
                                f"- LLaMA Suggestion: {groq_suggestions.get('llama', 'N/A')}\n"
                                f"- Gemma Suggestion: {groq_suggestions.get('gemma', 'N/A')}\n"
                                f"- Mistral Saba Suggestion: {groq_suggestions.get('mistral-saba', 'N/A')}"
                            ),
                            **user_params
                        }
                        merge_result_obj = await llm_gemini.ainvoke(
                            merge_prompt_template.format(**merge_input_kwargs),
                            config={
                                "callbacks": [SafeTracer()],
                                "configurable": {"session_id": session_id}
                            }
                        )
                        tool_output = merge_result_obj.content if isinstance(merge_result_obj, AIMessage) else str(merge_result_obj)
                        response_text = tool_output # This tool now always provides the final answer for diet generation
                        break # Diet generation is considered a terminal action for the agent loop
                    except Exception as e:
                        logging.error(f"❌ Merge error in generate_diet_plan tool: {e}", exc_info=True)
                        tool_output = "Error during diet plan generation (merge step)."
                        response_text = tool_output
                        break # Treat as final response if merge fails


                elif tool_name == "fetch_recipe":
                    # Removed 'break' here. This tool's output will now be added to scratchpad,
                    # and the agent can decide next if other tools are needed.
                    tool_output = await tool_fetch_recipe(tool_input.get("recipe_name", "unknown"))
                    # The response_text is set below, after the agent loop, by the orchestrator's final_answer
                    # We just capture the tool_output here for the scratchpad.

                elif tool_name == "lookup_nutrition_facts":
                    # Removed 'break' here. Similar to fetch_recipe, its output is for scratchpad.
                    tool_output = await tool_lookup_nutrition_facts(tool_input.get("food_item", "unknown"))
                    # The response_text is set below, after the agent loop, by the orchestrator's final_answer

                else:
                    tool_output = f"Error: Unknown tool '{tool_name}' requested by agent."
                    logging.warning(tool_output)
                    response_text = tool_output
                    break # Unknown tool implies an unrecoverable error for this turn

            except Exception as e:
                tool_output = f"Error executing tool '{tool_name}': {e}"
                logging.error(tool_output, exc_info=True)
                response_text = tool_output
                break # Break on critical tool execution error
            
            # Add the executed tool and its output to the scratchpad
            agent_scratchpad.append({
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output
            })

        else: # This 'else' block executes if the loop completes without a 'break' (i.e., no final_answer)
            if response_text == "I'm sorry, I encountered an internal issue and cannot respond right now. Please try again.":
                response_text = "I couldn't finalize my response after several attempts. Please try rephrasing your request."
            logging.warning(f"Agent loop finished without explicit final answer for session {session_id}. Final response: '{response_text[:100]}'")

    except ValidationError as e:
        logging.error(f"❌ Pydantic validation error in agent decision: {e}", exc_info=True)
        response_text = "I received an invalid instruction from my internal system. Please try again."
    except Exception as e:
        logging.error(f"❌ Global error in /chat endpoint for session {session_id}: {e}", exc_info=True)
        response_text = "I'm experiencing a technical issue and cannot respond at the moment. Please try again later."

    # Detect sentiment (run outside the loop, as a final logging step)
    sentiment = detect_sentiment(llm_orchestrator, user_query)
    logging.info(f"Sentiment for query '{user_query}': {sentiment}")

    # Add user query and AI response to session history
    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(response_text)

    # Log to Google Sheet (if enabled)
    try:
        if sheet_enabled and sheet:
            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Timestamp
                session_id,                                    # Session ID
                user_query,                                    # User Query
                response_text,                                 # AI Response
                sentiment                                      # Detected Sentiment
            ])
            logging.info("📝 Logged query, response, and sentiment to Google Sheet.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheet logging failed for session {session_id}: {e}", exc_info=True)

    return JSONResponse(content={"answer": response_text, "session_id": session_id})

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "✅ Indian Diet Recommendation API is running. Use POST /chat to interact."}

if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting FastAPI application locally...")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)
