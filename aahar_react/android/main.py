# fastapi_app.py - Corrected and Refined Version
import os
import json
import logging
import zipfile
import requests
import string
import re
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from fuzzywuzzy import fuzz, process

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Langchain and Google Generative AI imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import StringPromptValue

# --- Global Variables for Nutrition Dataset ---
nutrition_data: List[Dict[str, Any]] = []
nutrition_df: Optional[pd.DataFrame] = None

# --- Nutrition Dataset Loading ---
def load_nutrition_dataset(file_path: str = "nutrition_data.json"):
    """
    Load the nutrition dataset from JSON file and prepare it for efficient querying.
    """
    global nutrition_data, nutrition_df

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.warning(f"⚠️ Nutrition dataset file '{file_path}' not found. Using fallback data.")
            create_fallback_nutrition_data()
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            nutrition_data = json.load(f)

        # Convert to pandas DataFrame for easier querying
        nutrition_df = pd.DataFrame(nutrition_data)

        # Clean and standardize data
        if not nutrition_df.empty:
            # Convert numeric columns to proper types
            numeric_columns = ['Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Sugar (g)',
                               'Fat (g)', 'Fiber (g)', 'Sodium (mg)']

            for col in numeric_columns:
                if col in nutrition_df.columns:
                    nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce')

            # Create searchable text for fuzzy matching
            nutrition_df['searchable_text'] = (
                nutrition_df['Dish Name'].astype(str).str.lower() + ' ' +
                nutrition_df['Category'].astype(str).str.lower()
            )

        logging.info(f"✅ Loaded {len(nutrition_data)} nutrition records from {file_path}")

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON parsing error in nutrition dataset: {e}")
        create_fallback_nutrition_data()
    except Exception as e:
        logging.error(f"❌ Error loading nutrition dataset: {e}")
        create_fallback_nutrition_data()

def create_fallback_nutrition_data():
    """
    Create basic fallback nutrition data if the JSON file is not available.
    """
    global nutrition_data, nutrition_df

    fallback_data = [
        {
            "Category": "Breads & Roti",
            "Dish Name": "Plain Roti / Chapati (Whole Wheat)",
            "Region": "Pan-India",
            "Serving Size": "1 medium",
            "Calories (kcal)": 95,
            "Protein (g)": 3,
            "Carbs (g)": 18,
            "Sugar (g)": 0,
            "Fat (g)": 1,
            "Fiber (g)": 3,
            "Sodium (mg)": 150,
            "Key Vitamins & Minerals": "Iron, Magnesium, B-Vitamins"
        },
        {
            "Category": "Rice & Grains",
            "Dish Name": "Cooked Rice (White)",
            "Region": "Pan-India",
            "Serving Size": "1 cup",
            "Calories (kcal)": 205,
            "Protein (g)": 4.3,
            "Carbs (g)": 45,
            "Sugar (g)": 0.1,
            "Fat (g)": 0.4,
            "Fiber (g)": 0.6,
            "Sodium (mg)": 2,
            "Key Vitamins & Minerals": "Manganese, Selenium"
        },
        {
            "Category": "Legumes & Dal",
            "Dish Name": "Cooked Lentils (Mixed Dal)",
            "Region": "Pan-India",
            "Serving Size": "1 cup",
            "Calories (kcal)": 230,
            "Protein (g)": 18,
            "Carbs (g)": 40,
            "Sugar (g)": 4,
            "Fat (g)": 0.8,
            "Fiber (g)": 16,
            "Sodium (mg)": 4,
            "Key Vitamins & Minerals": "Iron, Folate, Potassium, Magnesium"
        }
    ]

    nutrition_data = fallback_data
    nutrition_df = pd.DataFrame(fallback_data)

    # Add searchable text
    if not nutrition_df.empty:
        nutrition_df['searchable_text'] = (
            nutrition_df['Dish Name'].astype(str).str.lower() + ' ' +
            nutrition_df['Category'].astype(str).str.lower()
        )

    logging.info(f"✅ Created fallback nutrition dataset with {len(fallback_data)} records")

def search_nutrition_data(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search the nutrition dataset using fuzzy matching for food items with improved accuracy.
    """
    if nutrition_df is None or nutrition_df.empty:
        return []

    query_lower = query.lower().strip()

    # 1. High-priority: Exact match on 'Dish Name'
    exact_matches = nutrition_df[nutrition_df['Dish Name'].str.lower() == query_lower]
    if not exact_matches.empty:
        return exact_matches.head(limit).to_dict('records')

    # 2. Secondary priority: Contains match on 'Dish Name'
    contains_matches = nutrition_df[
        nutrition_df['Dish Name'].str.lower().str.contains(query_lower, na=False, regex=False)
    ]
    if not contains_matches.empty:
        # Sort by length to prefer shorter, more exact matches
        contains_matches = contains_matches.copy()
        contains_matches['match_len'] = contains_matches['Dish Name'].str.len()
        return contains_matches.sort_values('match_len').head(limit).drop(columns=['match_len']).to_dict('records')

    # 3. Fuzzy matching on 'searchable_text' with a high threshold
    try:
        choices = nutrition_df['searchable_text'].tolist()
        # Use a more robust scorer and a higher threshold to avoid incorrect matches
        matches = process.extract(query_lower, choices, limit=limit, scorer=fuzz.token_set_ratio)

        # Filter matches with a high similarity score (e.g., > 85)
        good_matches = [match for match in matches if match[1] > 85]

        if good_matches:
            matched_indices = [choices.index(match[0]) for match in good_matches]
            return nutrition_df.iloc[matched_indices].to_dict('records')

    except Exception as e:
        logging.error(f"Error in fuzzy matching: {e}")

    # If all other methods fail, return an empty list. Avoids overly broad category matches.
    return []


def get_regional_nutrition_suggestions(region: str, dietary_type: str, goal: str) -> List[Dict[str, Any]]:
    """
    Get nutrition suggestions based on region, dietary preferences, and goals.
    """
    if nutrition_df is None or nutrition_df.empty:
        return []

    # Filter by region if specified
    filtered_df = nutrition_df.copy()

    if region and region != "Indian":
        region_matches = filtered_df[
            filtered_df['Region'].str.lower().str.contains(region.lower(), na=False, regex=False)
        ]
        if not region_matches.empty:
            filtered_df = region_matches

    # Filter by dietary type
    if dietary_type == "vegan":
        # Exclude dairy and animal products
        vegan_categories = ['Breads & Roti', 'Rice & Grains', 'Legumes & Dal',
                            'Vegetables', 'Fruits', 'Nuts & Seeds']
        filtered_df = filtered_df[
            filtered_df['Category'].str.contains('|'.join(vegan_categories), case=False, na=False)
        ]
    elif dietary_type == "vegetarian":
        # Include vegetarian items, exclude meat/fish
        non_veg_keywords = ['chicken', 'fish', 'mutton', 'beef', 'pork', 'egg']
        for keyword in non_veg_keywords:
            filtered_df = filtered_df[
                ~filtered_df['Dish Name'].str.lower().str.contains(keyword, na=False, regex=False)
            ]

    # Sort by goal
    if goal == "weight loss":
        # Prefer low-calorie, high-fiber foods
        filtered_df = filtered_df.sort_values(['Calories (kcal)', 'Fiber (g)'],
                                              ascending=[True, False])
    elif goal == "weight gain":
        # Prefer high-calorie, high-protein foods
        filtered_df = filtered_df.sort_values(['Calories (kcal)', 'Protein (g)'],
                                              ascending=[False, False])
    else:
        # General diet - balanced approach
        filtered_df = filtered_df.sort_values('Fiber (g)', ascending=False)

    return filtered_df.head(10).to_dict('records')

def format_nutrition_info(nutrition_record: Dict[str, Any]) -> str:
    """
    Format a single nutrition record into a readable string.
    """
    try:
        dish_name = nutrition_record.get('Dish Name', 'Unknown Dish')
        category = nutrition_record.get('Category', 'Unknown Category')
        region = nutrition_record.get('Region', 'Unknown Region')
        serving_size = nutrition_record.get('Serving Size', 'Unknown serving')

        calories = nutrition_record.get('Calories (kcal)', 'N/A')
        protein = nutrition_record.get('Protein (g)', 'N/A')
        carbs = nutrition_record.get('Carbs (g)', 'N/A')
        fat = nutrition_record.get('Fat (g)', 'N/A')
        fiber = nutrition_record.get('Fiber (g)', 'N/A')

        vitamins = nutrition_record.get('Key Vitamins & Minerals', 'N/A')

        return f"""
**{dish_name}** ({category}, {region})
- Serving Size: {serving_size}
- Calories: {calories} kcal
- Protein: {protein}g | Carbs: {carbs}g | Fat: {fat}g | Fiber: {fiber}g
- Key Nutrients: {vitamins}
        """.strip()
    except Exception as e:
        logging.error(f"Error formatting nutrition info: {e}")
        return f"Error formatting nutrition information for {nutrition_record.get('Dish Name', 'Unknown')}"

# --- Consolidated: Custom Callback for LangChain ---
class SafeTracer(BaseCallbackHandler):
    """
    A custom LangChain callback handler to safely log chain outputs.
    """
    def on_chain_end(self, outputs: Any, **kwargs):
        try:
            if isinstance(outputs, (AIMessage, HumanMessage, SystemMessage)):
                logging.info(f"🔁 Chain ended. Message type: {type(outputs).__name__}, content snippet: {outputs.content[:100]}...")
            elif isinstance(outputs, StringPromptValue):
                logging.info(f"🔁 Chain ended. PromptValue content snippet: {outputs.text[:100]}...")
            elif isinstance(outputs, dict):
                if "answer" in outputs:
                    logging.info(f"🔁 Chain ended. Answer snippet: {outputs['answer'][:100]}...")
                elif "output" in outputs:
                    logging.info(f"🔁 Chain ended. Output snippet: {outputs['output'][:100]}...")
                elif "text" in outputs:
                    logging.info(f"🔁 Chain ended. Text output snippet: {outputs['text'][:100]}...")
                else:
                    logging.info(f"🔁 Chain ended. Output (type: {type(outputs)}, content snippet): {str(outputs)[:100]}...")
            elif isinstance(outputs, str):
                logging.info(f"🔁 Chain ended. String output snippet: {outputs[:100]}...")
            else:
                logging.info(f"🔁 Chain ended. Output (type: {type(outputs)}, content snippet): {str(outputs)[:100]}...")
        except Exception as e:
            logging.error(f"❌ Error in on_chain_end callback: {e}")

# --- Consolidated: Vector Database Setup & Download ---
def download_and_extract_db_for_app():
    """
    Downloads and extracts a prebuilt ChromaDB from a HuggingFace URL.
    """
    url = "https://huggingface.co/datasets/Dyno1307/chromadb-diet/resolve/main/db.zip"
    zip_path = "/tmp/db.zip"
    extract_path = "/tmp/chroma_db"

    os.makedirs(extract_path, exist_ok=True)

    if os.path.exists(os.path.join(extract_path, "index")):
        logging.info("✅ Chroma DB already exists, skipping download.")
        return

    try:
        logging.info("⬇️ Downloading Chroma DB zip from HuggingFace...")
        with requests.get(url, stream=True, timeout=120) as r:
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
            raise EnvironmentError("GEMINI_API_KEY not set in environment variables.")

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
            logging.warning(f"⚠️ Could not count documents in Vector DB: {e}")

        logging.info("✅ Chroma DB initialized successfully.")
        return db, embedding

    except Exception as e:
        logging.exception("❌ Vector DB setup failed.")
        raise

# --- Consolidated: Groq Integration ---
def cached_groq_answers(query: str, groq_api_key: str, dietary_type: str, goal: str, region: str) -> dict:
    """
    Fetches diet suggestions from multiple Groq models in parallel.
    """
    logging.info(f"Fetching Groq answers for query: '{query}', pref: '{dietary_type}', goal: '{goal}', region: '{region}'")
    models = ["llama", "gemma", "mixtral"]
    results = {}
    if not groq_api_key:
        logging.warning("GROQ_API_KEY not available. Skipping Groq calls.")
        return {k: "Groq API key not available." for k in models}

    def _groq_diet_answer_single(model_name: str):
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
            groq_model_map = {
                "llama": "llama3-70b-8192",
                "gemma": "gemma2-9b-it",
                "mixtral": "mixtral-8x7b-32768"
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

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and data.get('choices') and data['choices'][0].get('message'):
                return data['choices'][0]['message']['content']

            return f"No suggestion from {actual_model_name}."

        except Exception as e:
            logging.error(f"Error from {model_name}: {e}")
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
        logging.info(f"Creating new Langchain session history for: {session_id}")
        llm_chains_session_store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history for: {session_id}")
    return llm_chains_session_store[session_id]

def define_rag_prompt_template():
    """Defines the prompt template for the RAG chain with nutrition data integration."""
    template_string = """
    You are an AI assistant specialized in Indian diet and nutrition created by Suprovo.
    Based on the following conversation history and the user's query, provide a simple, practical, and culturally relevant **{dietary_type}** food suggestion suitable for Indian users aiming for **{goal}**.
    If a specific region like **{region}** is mentioned or inferred, prioritize food suggestions from that region.

    You have access to detailed nutrition information from a comprehensive Indian food database.
    Use this information to provide accurate nutritional details and make informed recommendations.
    Focus on readily available ingredients and common Indian dietary patterns for the specified region.

    Be helpful, encouraging, and specific where possible.
    Use the chat history to understand the context of the user's current query and maintain continuity.
    Strictly adhere to the **{dietary_type}** and **{goal}** requirements, and the **{region}** preference if specified.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    Nutrition Data Context:
    {nutrition_context}

    User Query:
    {query}

    {dietary_type} {goal} Food Suggestion (Tailored for {region} Indian context):
    """
    return PromptTemplate(
        template=template_string,
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "context", "nutrition_context"]
    )

def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    """Sets up the enhanced RAG chain with nutrition data integration."""
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

        def get_nutrition_context(input_dict):
            """Get relevant nutrition data based on the query."""
            query = input_dict["query"]
            dietary_type = input_dict.get("dietary_type", "any")
            goal = input_dict.get("goal", "diet")
            region = input_dict.get("region", "Indian")

            # Search for specific foods mentioned in the query
            nutrition_matches = search_nutrition_data(query, limit=3)

            # Get regional suggestions
            regional_suggestions = get_regional_nutrition_suggestions(region, dietary_type, goal)

            nutrition_context = ""

            if nutrition_matches:
                nutrition_context += "Specific Nutrition Information:\n"
                for item in nutrition_matches[:3]:
                    nutrition_context += format_nutrition_info(item) + "\n\n"

            if regional_suggestions and len(regional_suggestions) > 0:
                nutrition_context += f"Recommended {dietary_type} foods for {goal} in {region} context:\n"
                for item in regional_suggestions[:5]:
                    nutrition_context += f"- {item.get('Dish Name', 'Unknown')} ({item.get('Calories (kcal)', 'N/A')} kcal, {item.get('Protein (g)', 'N/A')}g protein)\n"

            return nutrition_context.strip()

        qa_chain = (
            {
                "context": retrieve_and_log_context,
                "nutrition_context": get_nutrition_context,
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
        logging.info("Enhanced Retrieval QA Chain with nutrition data initialized successfully.")
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
    """Defines prompt templates for merging RAG and Groq outputs with nutrition data."""
    merge_prompt_default_template = """
    You are an AI assistant specialized in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.

    You have access to detailed nutrition information from a comprehensive database. Use this information to provide accurate nutritional details and calorie counts.

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}
    {nutrition_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. Use the nutrition data to provide accurate calorie, protein, and nutrient information.
    3. If the "Primary RAG Answer" is generic or insufficient, rely on "Additional Suggestions" and nutrition data.
    4. Combine information logically and seamlessly, without mentioning the source of each piece.
    5. Ensure the final plan is clear, actionable, culturally relevant, and nutritionally accurate.
    6. Include specific nutritional values where possible (calories, protein, etc.).

    Final {dietary_type} {goal} Food Suggestion/Diet Plan (Tailored for {region} Indian context):
    """

    merge_prompt_table_template = """
    You are an AI assistant specialized in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.
    **You MUST present the final diet plan as a clear markdown table. Include columns for Meal, Food Items, Serving Size, Calories, and Key Nutrients.**

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}
    {nutrition_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. Use the detailed nutrition data to provide accurate serving sizes, calorie counts, and nutrient information in the table.
    3. If the "Primary RAG Answer" is generic or insufficient, rely on "Additional Suggestions" and nutrition data.
    4. Combine information logically and seamlessly, without mentioning the source of each piece.
    5. Ensure the final plan is clear, actionable, culturally relevant, and nutritionally accurate.
    6. The table must include specific nutritional values from the database where available.

    Final {dietary_type} {goal} Diet Plan (Tailored for {region} Indian context, in markdown table format):
    """

    logging.info("Enhanced Merge Prompt templates with nutrition integration created.")
    return (
        PromptTemplate(template=merge_prompt_default_template,
                      input_variables=["rag_section", "additional_suggestions_section", "nutrition_section", "dietary_type", "goal", "region"]),
        PromptTemplate(template=merge_prompt_table_template,
                      input_variables=["rag_section", "additional_suggestions_section", "nutrition_section", "dietary_type", "goal", "region"])
    )

# --- Enhanced Query Analysis ---
def clean_query(query: str) -> str:
    """Strips punctuation and lowercases the query for consistent keyword matching."""
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

@lru_cache(maxsize=128)
def extract_diet_preference(query: str) -> str:
    """Extracts dietary preference from the query."""
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
    """Extracts diet goal from the query."""
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
    """Extracts regional preference for Indian diet."""
    q = query.lower()
    if "kolkata" in q or "bengali" in q:
        return "Bengali"
    if any(term in q for term in ["south indian", "tamil", "kannada", "telugu", "malayalam", "kanyakumari"]):
        return "South Indian"
    if any(term in q for term in ["north indian", "punjabi"]):
        return "North Indian"
    if any(term in q for term in ["west indian", "maharashtrian", "gujarati"]):
        return "West Indian"
    if any(term in q for term in ["east indian", "odisha", "oriya", "bhubaneswar", "cuttack", "angul"]):
        return "East Indian"
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

        if isinstance(response_obj, AIMessage):
            sentiment = response_obj.content.strip().lower()
        elif isinstance(response_obj, str):
            sentiment = response_obj.strip().lower()
        else:
            logging.warning(f"LLM returned unexpected type for sentiment: {type(response_obj)}. Defaulting to 'neutral'.")
            return "neutral"

        if sentiment in ["positive", "neutral", "negative"]:
            return sentiment
        else:
            logging.warning(f"LLM returned unexpected sentiment: '{sentiment}'. Defaulting to 'neutral'.")
            return "neutral"
    except Exception as e:
        logging.error(f"Error detecting sentiment: {e}", exc_info=True)
        return "neutral"

# --- Enhanced Tools with Nutrition Data ---
async def tool_fetch_recipe(recipe_name: str) -> str:
    """Enhanced recipe fetching with nutrition data integration."""
    logging.info(f"Executing tool: fetch_recipe for '{recipe_name}'")
    import asyncio
    await asyncio.sleep(0.5)

    # Search nutrition database for the recipe
    nutrition_matches = search_nutrition_data(recipe_name, limit=1)

    basic_recipe = ""
    if "dal makhani" in recipe_name.lower():
        basic_recipe = "Recipe for Dal Makhani: Ingredients - Black lentils, kidney beans, butter, cream, tomatoes, ginger-garlic paste. Steps - Soak overnight, boil lentils, prepare tempering, simmer with spices and cream. Serve hot with naan or rice."
    elif "paneer tikka" in recipe_name.lower():
        basic_recipe = "Recipe for Paneer Tikka: Ingredients - Paneer, yogurt, ginger-garlic paste, red chili powder, garam masala, bell peppers, onions. Steps - Cut paneer and vegetables, marinate with spices, skewer and grill/bake until golden."
    elif "chicken tikka masala" in recipe_name.lower():
        basic_recipe = "Recipe for Chicken Tikka Masala: Ingredients - Chicken, yogurt, ginger-garlic paste, spices, tomatoes, cream, onions. Steps - Marinate chicken, grill/bake, prepare rich tomato-cream sauce, combine and simmer."
    else:
        basic_recipe = f"Recipe for {recipe_name}: Detailed recipe unavailable, but typically involves fresh ingredients and traditional Indian cooking methods."

    # Add nutrition information if available
    if nutrition_matches:
        nutrition_info = format_nutrition_info(nutrition_matches[0])
        return f"{basic_recipe}\n\n**Nutrition Information:**\n{nutrition_info}"

    return basic_recipe

async def tool_lookup_nutrition_facts(food_item: str) -> str:
    """Enhanced nutrition lookup using the JSON dataset."""
    logging.info(f"Executing tool: lookup_nutrition_facts for '{food_item}'")
    import asyncio
    await asyncio.sleep(0.5)

    # Search the nutrition database first
    nutrition_matches = search_nutrition_data(food_item, limit=3)

    if nutrition_matches:
        result = f"**Detailed Nutrition Information for '{food_item}':**\n\n"
        for i, match in enumerate(nutrition_matches, 1):
            result += f"**Option {i}:** {format_nutrition_info(match)}\n\n"
        return result

    # Fallback to original logic for specific comparisons
    clean_food_item = food_item.lower().strip()

    if "non veg vs veg" in clean_food_item or "non-veg vs veg" in clean_food_item:
        return """
**Comparing Non-Vegetarian vs. Vegetarian Nutrition:**

**Non-Vegetarian (e.g., Chicken Breast - 100g cooked):**
- Calories: ~165 kcal
- Protein: ~31g (complete protein with all essential amino acids)
- Fat: ~3.6g (low in saturated fat if skinless)
- Carbs: 0g
- Key nutrients: B vitamins (B12, niacin), iron (heme), zinc, selenium

**Vegetarian Protein Sources:**

*Lentils (100g cooked):*
- Calories: ~116 kcal
- Protein: ~9g (incomplete, but becomes complete when paired with grains)
- Fat: ~0.4g (very low)
- Carbs: ~20g
- Key nutrients: Fiber (8g), folate, potassium, iron (non-heme), magnesium

*Paneer (100g):*
- Calories: ~265 kcal
- Protein: ~18g (complete protein)
- Fat: ~20g (higher in saturated fat)
- Carbs: ~1.2g
- Key nutrients: Calcium (208mg), phosphorus, Vitamin B12

**Summary:** Non-vegetarian options provide complete proteins and better iron/B12 bioavailability. Vegetarian diets excel in fiber, diverse micronutrients, and can be lower in saturated fat. Both can meet nutritional needs with proper planning.
        """.strip()

    # Generic fallback
    return f"Specific nutrition data for '{food_item}' not found in our database. For accurate nutrition information, please specify a common Indian food item or dish name."

async def tool_get_nutrition_comparison(food_items: List[str]) -> str:
    """New tool to compare nutrition between multiple food items."""
    logging.info(f"Executing tool: get_nutrition_comparison for {food_items}")

    if len(food_items) < 2:
        return "Please provide at least 2 food items for comparison."

    comparison_result = "**Nutrition Comparison:**\n\n"

    for food_item in food_items[:5]: # Limit to 5 items
        matches = search_nutrition_data(food_item, limit=1)
        if matches:
            comparison_result += format_nutrition_info(matches[0]) + "\n" + "="*50 + "\n"
        else:
            comparison_result += f"**{food_item}:** Nutrition data not available\n" + "="*50 + "\n"

    return comparison_result

# --- Agentic Orchestration Models ---
class AgentAction(BaseModel):
    """Represents the action the AI Agent decides to take."""
    thought: Optional[str] = Field(None, description="A brief thought process explaining the current decision.")
    tool_name: Optional[str] = Field(None, description="The name of the tool to use.")
    tool_input: Optional[Dict[str, Any]] = Field(None, description="Parameters for the selected tool.")
    final_answer: Optional[str] = Field(None, description="The final answer to the user's request.")

# --- Enhanced Orchestrator Prompt ---
ORCHESTRATOR_PROMPT_TEMPLATE = """
You are AAHAR, an intelligent AI agent specialized in Indian diet and nutrition with access to a comprehensive nutrition database.
Your goal is to assist users with diet-related queries by thinking step-by-step and providing accurate, data-driven answers.

You have access to a detailed nutrition database containing information about Indian foods including calories, protein, carbs, fats, fiber, and key vitamins/minerals.

Available Tools:
1. **handle_greeting**: Respond to simple greetings
2. **handle_identity**: Respond to identity questions
3. **reformat_diet_plan**: Reformat previous diet plans (only if there's a substantial previous AI response)
4. **generate_diet_plan**: Generate new diet suggestions using RAG, nutrition database, and Groq models
    - Input: `dietary_type`, `goal`, `region`, `wants_table`
5. **fetch_recipe**: Get recipes with integrated nutrition information
    - Input: `recipe_name`
6. **lookup_nutrition_facts**: Look up detailed nutrition facts from the database
    - Input: `food_item`
7. **get_nutrition_comparison**: Compare nutrition between multiple food items
    - Input: `food_items` (list of food names)
8. **get_weather_based_suggestion**: Weather-appropriate food suggestions
    - Input: `city`

**Current State:**
Chat History: {chat_history}
Current User Query: "{query}"
Agent Scratchpad: {agent_scratchpad}

**Decision Making:**
- If user asks for nutrition facts or comparisons, use lookup_nutrition_facts or get_nutrition_comparison
- If user asks for recipes, use fetch_recipe (will include nutrition data)
- For diet plans, use generate_diet_plan (enhanced with nutrition database)
- For weather-based suggestions, use get_weather_based_suggestion
- If you've executed a tool that answers the user's query, set final_answer and stop
- Always provide nutritionally accurate information using the database

Output JSON adhering to AgentAction model:
"""

ORCHESTRATOR_PROMPT = PromptTemplate(
    template=ORCHESTRATOR_PROMPT_TEMPLATE,
    input_variables=["chat_history", "query", "agent_scratchpad"]
)

# --- Weather Tool ---
def get_weather(city: str) -> Optional[Dict[str, Any]]:
    """Fetch current weather data for a given city using OpenWeatherMap."""
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "cbf74aade9b233c5dc5f99b1b49f7d50")

    if not OPENWEATHER_API_KEY:
        logging.error("❌ OPENWEATHER_API_KEY is not set.")
        return None

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            logging.error(f"Error fetching weather for {city}: {data.get('message', 'Unable to fetch weather')}")
            return None

        return {
            "city": city,
            "temperature": data['main']['temp'],
            "condition": data['weather'][0]['description'],
            "humidity": data['main']['humidity']
        }
    except Exception as e:
        logging.error(f"Error fetching weather for {city}: {e}", exc_info=True)
        return None

# --- FastAPI App Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "cbf74aade9b233c5dc5f99b1b49f7d50")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_secure_random_key_CHANGE_THIS_IN_PRODUCTION")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('langchain_chroma.base').setLevel(logging.WARNING)

# FastAPI App Initialization
app = FastAPI(
    title="Enhanced Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions and meal analysis using RAG, nutrition database, and LLMs.",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# Global variables for initialized components
llm_gemini: Optional[GoogleGenerativeAI] = None
llm_orchestrator: Optional[GoogleGenerativeAI] = None
db: Optional[Chroma] = None
rag_prompt: Optional[PromptTemplate] = None
qa_chain: Optional[Any] = None
conversational_qa_chain: Optional[Any] = None
merge_prompt_default: Optional[PromptTemplate] = None
merge_prompt_table: Optional[PromptTemplate] = None
orchestrator_chain: Optional[Any] = None
weather_suggestion_prompt: Optional[PromptTemplate] = None

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with nutrition dataset loading."""
    global llm_gemini, llm_orchestrator, db, rag_prompt, qa_chain, conversational_qa_chain, \
           merge_prompt_default, merge_prompt_table, orchestrator_chain, weather_suggestion_prompt

    try:
        # Load nutrition dataset first
        logging.info("📊 Loading nutrition dataset...")
        load_nutrition_dataset("nutrition_data.json")

        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is not set.")

        llm_gemini = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.5
        )
        llm_orchestrator = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1
        )
        logging.info("✅ Gemini LLMs initialized.")
    except Exception as e:
        logging.error(f"❌ Gemini LLM initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini LLM initialization failed.")

    try:
        download_and_extract_db_for_app()
        db, _ = setup_vector_database(chroma_db_directory="/tmp/chroma_db")
        logging.info("✅ Vector DB initialized.")
    except Exception as e:
        logging.error(f"❌ Vector DB init error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Vector DB initialization failed.")

    try:
        rag_prompt = define_rag_prompt_template()
        qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
        conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
        merge_prompt_default, merge_prompt_table = define_merge_prompt_templates()

        # Weather suggestion prompt
        weather_suggestion_prompt_template = """
        You are an AI assistant specialized in Indian diet and nutrition with access to detailed nutrition data.
        The user wants a diet suggestion for the city of **{city}**.
        Current weather: Temperature: **{temperature}°C**, Condition: **{condition}**, Humidity: **{humidity}%**.

        Based on this weather, provide a practical **{dietary_type}** food suggestion for **{goal}**.
        Use your nutrition database knowledge to suggest appropriate foods with calorie and nutrient information.
        For hot weather, suggest cooling foods. For cold/rainy weather, suggest warm, comforting foods.

        User's query: "{query}"

        Weather-Appropriate Food Suggestion with Nutrition Details:
        """
        weather_suggestion_prompt = PromptTemplate(
            template=weather_suggestion_prompt_template,
            input_variables=["city", "temperature", "condition", "humidity", "dietary_type", "goal", "query"]
        )

        def parse_agent_action_output(llm_output: Union[AIMessage, str]) -> AgentAction:
            """More robustly parse LLM output into AgentAction model."""
            content_str = llm_output.content if isinstance(llm_output, AIMessage) else str(llm_output)

            json_match = re.search(r"```json\n(.*?)\n```", content_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = content_str.strip()

            try:
                data = json.loads(json_str)
                # Ensure it's a dictionary before passing to the model
                if isinstance(data, dict):
                    return AgentAction(**data)
                else:
                    logging.error(f"Parsed JSON is not a dictionary. Type: {type(data)}")
                    return AgentAction(
                        thought="LLM output was valid JSON but not a JSON object.",
                        final_answer="An internal system error occurred due to an unexpected data format from the AI. Please try again."
                    )
            except (ValidationError, json.JSONDecodeError) as e:
                logging.error(f"Error parsing or validating LLM output: {e}. Raw JSON string: {json_str}")
                return AgentAction(
                    thought="Orchestrator returned invalid or malformed JSON.",
                    final_answer="An internal system error occurred while processing the AI's decision. Please try again."
                )

        orchestrator_chain = (
            ORCHESTRATOR_PROMPT
            | llm_orchestrator
            | RunnableLambda(parse_agent_action_output)
        )

        logging.info("✅ All components initialized successfully with nutrition data integration.")

    except Exception as e:
        logging.error(f"❌ Component setup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Component initialization failed.")

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default=None, description="Optional session ID")

# --- NEW: Models for Meal Analyzer ---
class MealAnalysisRequest(BaseModel):
    dish_names: List[str]

class MealAnalysisResponse(BaseModel):
    analysis: str
    totals: Dict[str, Union[float, int]]
    found_dishes: List[Dict[str, Any]]
    not_found_dishes: List[str]


# --- API Endpoints ---
@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    """Enhanced chat endpoint with nutrition database integration."""
    user_query = chat_request.query
    client_session_id = chat_request.session_id

    session_id = client_session_id or request.session.get("session_id") or f"session_{os.urandom(8).hex()}"
    request.session["session_id"] = session_id

    logging.info(f"📩 Query: '{user_query}' | Session: {session_id}")

    response_text = "I'm sorry, I encountered an internal issue. Please try again."

    chat_history_lc = get_session_history(session_id).messages
    formatted_chat_history = ""
    for msg in chat_history_lc:
        if isinstance(msg, HumanMessage):
            formatted_chat_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_chat_history += f"AI: {msg.content}\n"

    # Enhanced Agent Loop
    max_agent_iterations = 6
    agent_scratchpad: List[Dict[str, Any]] = []

    try:
        for i in range(max_agent_iterations):
            logging.info(f"🔄 Agent Iteration {i+1}/{max_agent_iterations}")

            scratchpad_str = "\n".join([
                f"Tool: {item.get('tool_name')}\nInput: {item.get('tool_input')}\nOutput: {item.get('tool_output')}"
                for item in agent_scratchpad
            ])

            orchestrator_decision: AgentAction = await orchestrator_chain.ainvoke({
                "query": user_query,
                "chat_history": formatted_chat_history,
                "agent_scratchpad": scratchpad_str
            }, config={
                "callbacks": [SafeTracer()],
                "configurable": {"session_id": session_id}
            })

            logging.info(f"✨ Decision (Iter {i+1}): Tool='{orchestrator_decision.tool_name}', Params={orchestrator_decision.tool_input}")

            if orchestrator_decision.final_answer:
                response_text = orchestrator_decision.final_answer
                logging.info(f"✅ Final answer on iteration {i+1}.")
                break

            tool_name = orchestrator_decision.tool_name
            tool_input = orchestrator_decision.tool_input if orchestrator_decision.tool_input is not None else {}
            tool_output = "Error: Tool execution failed."

            try:
                if tool_name == "handle_greeting":
                    response_text = "Namaste! I'm AAHAR, your AI nutrition assistant with access to a comprehensive Indian food database. How can I help you with healthy diet suggestions today?"
                    tool_output = response_text
                    break

                elif tool_name == "handle_identity":
                    response_text = "I am AAHAR, an AI assistant specialized in Indian diet and nutrition, created by Suprovo. I have access to a detailed nutrition database with information about Indian foods and their nutritional values."
                    tool_output = response_text
                    break

                elif tool_name == "reformat_diet_plan":
                    logging.info("Executing tool: reformat_diet_plan with nutrition data.")
                    last_ai_message_content = None
                    for msg in reversed(chat_history_lc):
                        if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 50:
                            last_ai_message_content = msg.content
                            break

                    if last_ai_message_content:
                        wants_table_flag = tool_input.get("wants_table", False)
                        merge_prompt_template = merge_prompt_table if wants_table_flag else merge_prompt_default

                        user_params = {
                            "dietary_type": tool_input.get("dietary_type", "any"),
                            "goal": tool_input.get("goal", "diet"),
                            "region": tool_input.get("region", "Indian"),
                        }
                        nutrition_suggestions = get_regional_nutrition_suggestions(
                            user_params["region"], user_params["dietary_type"], user_params["goal"]
                        )
                        nutrition_context = ""
                        if nutrition_suggestions:
                            nutrition_context = "Available Nutrition Data:\n"
                            for item in nutrition_suggestions[:5]:
                                nutrition_context += f"- {item.get('Dish Name', 'Unknown')} ({item.get('Calories (kcal)', 'N/A')} kcal)\n"

                        reformat_response_obj = await llm_gemini.ainvoke(
                            merge_prompt_template.format(
                                rag_section=f"Previous Answer to Reformat:\n{last_ai_message_content}",
                                additional_suggestions_section="",
                                nutrition_section=nutrition_context,
                                **user_params
                            ),
                            config={"callbacks": [SafeTracer()], "configurable": {"session_id": session_id}}
                        )
                        tool_output = reformat_response_obj.content if isinstance(reformat_response_obj, AIMessage) else str(reformat_response_obj)
                        response_text = tool_output
                        break
                    else:
                        tool_output = "No substantial previous diet plan found to reformat."
                        response_text = tool_output
                        break

                elif tool_name == "generate_diet_plan":
                    logging.info("Executing enhanced generate_diet_plan with nutrition database.")
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
                            **user_params
                        }, config={
                            "callbacks": [SafeTracer()],
                            "configurable": {"session_id": session_id}
                        })
                        rag_output_content = str(rag_result)
                    except Exception as e:
                        logging.error(f"❌ RAG error: {e}", exc_info=True)
                        rag_output_content = "Error retrieving from knowledge base."

                    groq_suggestions = {}
                    if GROQ_API_KEY:
                        try:
                            groq_suggestions = cached_groq_answers(
                                query=user_query,
                                groq_api_key=GROQ_API_KEY,
                                **user_params
                            )
                        except Exception as e:
                            logging.error(f"❌ Groq error: {e}", exc_info=True)
                            groq_suggestions = {"llama": "Error", "gemma": "Error", "mixtral": "Error"}

                    nutrition_suggestions = get_regional_nutrition_suggestions(
                        user_params["region"], user_params["dietary_type"], user_params["goal"]
                    )
                    nutrition_context = ""
                    if nutrition_suggestions:
                        nutrition_context = "Detailed Nutrition Database Information:\n"
                        for item in nutrition_suggestions[:8]:
                            nutrition_context += format_nutrition_info(item) + "\n"

                    merge_prompt_template = merge_prompt_table if wants_table_flag else merge_prompt_default
                    try:
                        merge_result_obj = await llm_gemini.ainvoke(
                            merge_prompt_template.format(
                                rag_section=f"Primary RAG Answer:\n{rag_output_content}",
                                additional_suggestions_section=(
                                    f"- LLaMA: {groq_suggestions.get('llama', 'N/A')}\n"
                                    f"- Gemma: {groq_suggestions.get('gemma', 'N/A')}\n"
                                    f"- Mixtral: {groq_suggestions.get('mixtral', 'N/A')}"
                                ),
                                nutrition_section=nutrition_context,
                                **user_params
                            ),
                            config={"callbacks": [SafeTracer()], "configurable": {"session_id": session_id}}
                        )
                        tool_output = merge_result_obj.content if isinstance(merge_result_obj, AIMessage) else str(merge_result_obj)
                        response_text = tool_output
                        break
                    except Exception as e:
                        logging.error(f"❌ Merge error: {e}", exc_info=True)
                        tool_output = "Error generating comprehensive diet plan."
                        response_text = tool_output
                        break

                elif tool_name == "fetch_recipe":
                    tool_output = await tool_fetch_recipe(tool_input.get("recipe_name", "unknown"))
                    response_text = tool_output
                    break

                elif tool_name == "lookup_nutrition_facts":
                    tool_output = await tool_lookup_nutrition_facts(tool_input.get("food_item", "unknown"))
                    response_text = tool_output
                    break

                elif tool_name == "get_nutrition_comparison":
                    tool_output = await tool_get_nutrition_comparison(tool_input.get("food_items", []))
                    response_text = tool_output
                    break

                elif tool_name == "get_weather_based_suggestion":
                    logging.info("Executing weather-based suggestion with nutrition data.")
                    city = tool_input.get("city")
                    if not city:
                        tool_output = "City not provided for weather suggestion."
                        response_text = tool_output
                        break

                    weather_data = get_weather(city)
                    if not weather_data:
                        tool_output = f"Couldn't retrieve weather for {city}. Please check the city name."
                        response_text = tool_output
                        break

                    dietary_type = extract_diet_preference(user_query)
                    goal = extract_diet_goal(user_query)

                    suggestion_result_obj = await llm_gemini.ainvoke(
                        weather_suggestion_prompt.format(
                            **weather_data,
                            dietary_type=dietary_type,
                            goal=goal,
                            query=user_query
                        ),
                        config={"callbacks": [SafeTracer()]}
                    )
                    tool_output = suggestion_result_obj.content if isinstance(suggestion_result_obj, AIMessage) else str(suggestion_result_obj)
                    response_text = tool_output
                    break

                else:
                    tool_output = f"Unknown tool '{tool_name}' requested."
                    logging.warning(tool_output)
                    response_text = tool_output
                    break

            except Exception as e:
                tool_output = f"Error executing tool '{tool_name}': {e}"
                logging.error(tool_output, exc_info=True)
                response_text = tool_output
                break

            agent_scratchpad.append({
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output
            })

        else:
            if response_text == "I'm sorry, I encountered an internal issue. Please try again.":
                response_text = "I couldn't finalize my response after several attempts. Please try rephrasing your request."
            logging.warning(f"Agent loop finished without explicit final answer for session {session_id}.")

    except ValidationError as e:
        logging.error(f"❌ Pydantic validation error: {e}", exc_info=True)
        response_text = "I received an invalid instruction from my internal system. Please try again."
    except Exception as e:
        logging.error(f"❌ Global error in /chat endpoint for session {session_id}: {e}", exc_info=True)
        response_text = "I'm experiencing a technical issue. Please try again later."

    # Add messages to session history
    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(response_text)

    return JSONResponse(content={"answer": response_text, "session_id": session_id})


# <<<< --- START: NEW INTEGRATED CODE --- >>>>

MEAL_ANALYSIS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["dish_list", "totals_summary", "not_found_list"],
    template="""
You are an expert AI nutritionist. A user has provided a list of Indian dishes they have eaten in a meal.
Based on the nutritional data provided, give a concise and helpful analysis of the meal.

**Meal Composition:**
{dish_list}

**Identified Items' Total Nutritional Summary:**
{totals_summary}

**Items Not Found in Database:**
{not_found_list}

**Your Task:**
Provide a brief, helpful analysis of this meal in 3-5 clear sentences.
1.  Comment on the overall balance (e.g., "This meal is well-balanced in protein and carbs...", "This meal is high in fat...").
2.  Mention its caloric content (e.g., "It's a high-calorie meal suitable for weight gain...", "This is a light meal...").
3.  Point out any notable health aspects (e.g., "It offers a good amount of fiber...", "Be mindful of the high sodium content...").
4.  Conclude with a summary of its suitability (e.g., "Overall, a great post-workout recovery meal.", "A decent choice for a light lunch, but could use more protein.").
5.  If any dishes were not found, briefly mention that the analysis is based only on the identified items. Do not lecture the user.

**Meal Analysis:**
"""
)

@app.post("/analyze-meal", response_model=MealAnalysisResponse, tags=["Meal Analysis"])
async def analyze_meal(meal_request: MealAnalysisRequest):
    """
    Analyzes a list of dish names, calculates total nutrition, and provides an AI-driven summary.
    """
    if not llm_gemini:
        raise HTTPException(status_code=503, detail="LLM service is not available.")
    if not meal_request.dish_names:
        raise HTTPException(status_code=400, detail="The 'dish_names' list cannot be empty.")

    logging.info(f"🔬 Analyzing meal with dishes: {meal_request.dish_names}")

    numeric_columns = ['Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Sugar (g)', 'Fat (g)', 'Fiber (g)', 'Sodium (mg)']
    totals = {col: 0.0 for col in numeric_columns}
    found_dishes_data = []
    not_found_dishes_names = []

    for dish_name in meal_request.dish_names:
        # Use the corrected search function to find the best match
        search_results = search_nutrition_data(dish_name, limit=1)
        if search_results:
            match = search_results[0]
            found_dishes_data.append(match)
            # Aggregate the nutritional values
            for col in numeric_columns:
                value = match.get(col, 0)
                if pd.notna(value) and isinstance(value, (int, float)):
                    totals[col] += value
        else:
            not_found_dishes_names.append(dish_name)

    if not found_dishes_data:
        return MealAnalysisResponse(
            analysis="No dishes from your list were found in our database. We are unable to provide an analysis.",
            totals={},
            found_dishes=[],
            not_found_dishes=not_found_dishes_names
        )

    # Prepare context for the LLM
    dish_list_str = "\n".join([f"- {d.get('Dish Name', 'Unknown')}" for d in found_dishes_data])
    totals_summary_str = json.dumps({k: round(v, 2) for k, v in totals.items()}, indent=2)
    not_found_str = ", ".join(not_found_dishes_names) if not_found_dishes_names else "None"

    # Generate the AI analysis
    try:
        analysis_prompt = MEAL_ANALYSIS_PROMPT_TEMPLATE.format(
            dish_list=dish_list_str,
            totals_summary=totals_summary_str,
            not_found_list=not_found_str
        )
        ai_response = await llm_gemini.ainvoke(analysis_prompt, config={"callbacks": [SafeTracer()]})
        ai_analysis = ai_response.content if isinstance(ai_response, AIMessage) else str(ai_response)
    except Exception as e:
        logging.error(f"❌ LLM error during meal analysis: {e}", exc_info=True)
        ai_analysis = "An error occurred while generating the AI analysis for this meal."

    # Return the full response object
    return MealAnalysisResponse(
        analysis=ai_analysis.strip(),
        totals={k: round(v, 2) for k, v in totals.items()},
        found_dishes=found_dishes_data,
        not_found_dishes=not_found_dishes_names
    )

# <<<< --- END: NEW INTEGRATED CODE --- >>>>


@app.get("/")
async def root():
    """Root endpoint with enhanced API information."""
    return {
        "message": "✅ Enhanced Indian Diet Recommendation API with Nutrition Database is running.",
        "features": [
            "RAG-based diet suggestions",
            "Comprehensive nutrition database integration",
            "Multi-model LLM responses (Groq + Gemini)",
            "Weather-based food suggestions",
            "Nutrition facts lookup and comparison",
            "Recipe suggestions with nutrition info",
            "Regional cuisine preferences",
            "Full meal nutritional analysis"
        ],
        "usage": "Use POST /chat for conversational AI or POST /analyze-meal for meal analysis.",
        "nutrition_database_records": len(nutrition_data) if nutrition_data else 0
    }

@app.get("/nutrition/search/{food_name}", tags=["Nutrition Database"])
async def search_nutrition_endpoint(food_name: str, limit: int = 5):
    """Direct endpoint to search nutrition database."""
    try:
        results = search_nutrition_data(food_name, limit=limit)
        return {
            "query": food_name,
            "results_found": len(results),
            "results": results
        }
    except Exception as e:
        logging.error(f"Error in nutrition search endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error searching nutrition database")

@app.get("/nutrition/categories", tags=["Nutrition Database"])
async def get_nutrition_categories():
    """Get all available food categories in the nutrition database."""
    try:
        if nutrition_df is not None and not nutrition_df.empty:
            categories = nutrition_df['Category'].unique().tolist()
            category_counts = nutrition_df['Category'].value_counts().to_dict()
            return {
                "total_categories": len(categories),
                "categories": sorted(categories),
                "items_per_category": category_counts
            }
        else:
            return {"message": "Nutrition database not loaded or empty"}
    except Exception as e:
        logging.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving categories")

@app.get("/nutrition/dishes-by-category", tags=["Nutrition Database"])
async def get_dishes_by_category(category: str, limit: int = 50):
    """Get all dishes for a specific category using a query parameter."""
    try:
        if nutrition_df is not None and not nutrition_df.empty:
            # Case-insensitive match for the category
            category_matches = nutrition_df[nutrition_df['Category'].str.lower() == category.lower()]
            if not category_matches.empty:
                return category_matches.head(limit).to_dict('records')
            else:
                raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")
        else:
            raise HTTPException(status_code=503, detail="Nutrition database not loaded or empty")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting dishes for category {category}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving dishes for category")



@app.get("/nutrition/regional/{region}", tags=["Nutrition Database"])
async def get_regional_foods(region: str, dietary_type: str = "any", goal: str = "diet", limit: int = 10):
    """Get nutrition suggestions for a specific region."""
    try:
        results = get_regional_nutrition_suggestions(region, dietary_type, goal)
        return {
            "region": region,
            "dietary_type": dietary_type,
            "goal": goal,
            "suggestions_found": len(results),
            "suggestions": results[:limit]
        }
    except Exception as e:
        logging.error(f"Error getting regional foods: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving regional foods")

@app.post("/nutrition/compare", tags=["Nutrition Database"])
async def compare_nutrition_endpoint(food_items: List[str]):
    """Compare nutrition between multiple food items."""
    try:
        if len(food_items) < 2:
            raise HTTPException(status_code=400, detail="At least 2 food items required for comparison")

        comparison_data = []
        for food_item in food_items[:5]: # Limit to 5 items
            matches = search_nutrition_data(food_item, limit=1)
            if matches:
                comparison_data.append(matches[0])
            else:
                comparison_data.append({
                    "Dish Name": food_item,
                    "error": "Not found in database"
                })

        return {
            "comparison": comparison_data,
            "items_compared": len(comparison_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in nutrition comparison endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error comparing nutrition data")

@app.get("/health", tags=["Utilities"])
async def health_check():
    """Enhanced health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "nutrition_database": len(nutrition_data) > 0 if nutrition_data else False,
            "vector_database": db is not None,
            "llm_gemini": llm_gemini is not None,
            "llm_orchestrator": llm_orchestrator is not None,
            "groq_api": bool(GROQ_API_KEY),
            "weather_api": bool(OPENWEATHER_API_KEY)
        },
        "database_stats": {
            "nutrition_records": len(nutrition_data) if nutrition_data else 0,
            "active_sessions": len(llm_chains_session_store)
        }
    }

    # Check if critical components are working
    critical_components = ["nutrition_database", "vector_database", "llm_gemini", "llm_orchestrator"]
    if not all(health_status["components"][comp] for comp in critical_components):
        health_status["status"] = "degraded"

    return health_status

# --- Additional Utility Endpoints ---
@app.post("/nutrition/upload", tags=["Utilities"])
async def upload_nutrition_data(file_content: str):
    """
    Endpoint to update nutrition database with new JSON content.
    Note: This is for development/testing. In production, you'd want proper authentication.
    """
    try:
        new_data = json.loads(file_content)
        global nutrition_data, nutrition_df

        # Validate the structure of the new data
        if isinstance(new_data, list) and len(new_data) > 0:
            required_fields = ["Dish Name", "Category", "Calories (kcal)", "Protein (g)"]
            if all(field in new_data[0] for field in required_fields):
                nutrition_data = new_data
                nutrition_df = pd.DataFrame(nutrition_data)

                # Re-process the data
                numeric_columns = ['Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Sugar (g)',
                                   'Fat (g)', 'Fiber (g)', 'Sodium (mg)']
                for col in numeric_columns:
                    if col in nutrition_df.columns:
                        nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce')

                nutrition_df['searchable_text'] = (
                    nutrition_df['Dish Name'].astype(str).str.lower() + ' ' +
                    nutrition_df['Category'].astype(str).str.lower()
                )

                logging.info(f"✅ Nutrition database updated with {len(nutrition_data)} records")
                return {
                    "status": "success",
                    "records_loaded": len(nutrition_data),
                    "message": "Nutrition database updated successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Invalid data structure. Required fields: {required_fields}"
                }
        else:
            return {
                "status": "error",
                "message": "Data must be a non-empty list of nutrition records"
            }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logging.error(f"Error updating nutrition database: {e}")
        raise HTTPException(status_code=500, detail="Error updating nutrition database")

@app.get("/analytics/popular-queries", tags=["Utilities"])
async def get_popular_queries():
    """Get analytics about popular queries (simplified version)."""
    try:
        # This would typically come from your logging/analytics system
        # For now, return session statistics
        return {
            "active_sessions": len(llm_chains_session_store),
            "total_nutrition_records": len(nutrition_data) if nutrition_data else 0,
            "database_categories": len(nutrition_df['Category'].unique()) if nutrition_df is not None else 0,
            "note": "Detailed analytics would require persistent storage and proper tracking"
        }
    except Exception as e:
        logging.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analytics")

# --- Run the application ---
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting Enhanced FastAPI application with Nutrition Database and Meal Analyzer...")
    # The test script uses port 10000, so we default to it here.
    uvicorn.run("fastapi_app5:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)