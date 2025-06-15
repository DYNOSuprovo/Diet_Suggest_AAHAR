# query_analysis.py

import string
import re
import logging
from functools import lru_cache
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Core keyword sets
GREETINGS = {
    "hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam",
    "good morning", "good afternoon", "good evening"
}

TASK_KEYWORDS = {
    "diet", "plan", "weight", "gain", "loss", "table", "format", "chart", "show", "give",
    "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian",
    "south", "north", "east", "west", "india", "bengali", "punjabi", "maharashtrian",
    "gujarati", "tamil", "kannada", "telugu", "malayalam", "kanyakumari",
    "odisha", "oriya", "bhubaneswar", "cuttack", "angul", "muscle", "mass",
    "healthy", "unhealthy", "calorie", "protein", "fiber", "carb", "fat",
    "diabetes", "bp", "sugar", "pressure", "cholesterol"
}

FORMATTING_KEYWORDS = {
    "table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate",
    "structured", "organized"
}

FILLER_WORDS = {
    "in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you",
    "the", "for", "my", "to", "with", "want", "now", "like", "need"
}

REGION_KEYWORDS = [
    "south indian", "north indian", "east indian", "west indian", "bengali", "punjabi",
    "maharashtrian", "gujarati", "tamil", "kannada", "telugu", "malayalam", "odisha",
    "oriya", "kanyakumari", "bhubaneswar", "cuttack", "angul"
]

DISEASE_KEYWORDS = {
    "diabetes": ["diabetes", "sugar"],
    "blood pressure": ["bp", "blood pressure", "pressure"],
    "cholesterol": ["cholesterol", "lipid"]
}

def clean_query(query: str) -> str:
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

@lru_cache(maxsize=128)
def is_greeting(query: str) -> bool:
    if not query:
        return False
    cleaned = clean_query(query)
    words = cleaned.split()
    contains_task = any(keyword in cleaned for keyword in TASK_KEYWORDS)
    return cleaned in GREETINGS and len(words) <= 3 and not contains_task

@lru_cache(maxsize=128)
def is_generic_query(query: str) -> bool:
    q = clean_query(query)
    words = q.split()
    if len(words) <= 3 and not any(k in q for k in TASK_KEYWORDS):
        return True
    return False

@lru_cache(maxsize=128)
def is_formatting_request(query: str) -> bool:
    if not query:
        return False
    cleaned = clean_query(query)
    words = cleaned.split()
    if not any(k in cleaned for k in FORMATTING_KEYWORDS):
        return False
    non_formatting = sum(1 for w in words if w not in FORMATTING_KEYWORDS and w not in FILLER_WORDS)
    if len(words) <= 6 and non_formatting <= 1:
        logging.info(f"⚙️ Formatting request (short): {query}")
        return True
    if all(w in FORMATTING_KEYWORDS or w in FILLER_WORDS for w in words):
        logging.info(f"⚙️ Formatting request (only formatting words): {query}")
        return True
    return False

@lru_cache(maxsize=128)
def contains_table_request(query: str) -> bool:
    q = clean_query(query)
    return any(k in q for k in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])

@lru_cache(maxsize=128)
def extract_diet_preference(query: str) -> str:
    q = clean_query(query)
    if any(x in q for x in ["non-veg", "non veg", "nonvegetarian"]):
        return "non-vegetarian"
    if "vegan" in q:
        return "vegan"
    if "veg" in q or "vegetarian" in q:
        return "vegetarian"
    return "any"

@lru_cache(maxsize=128)
def extract_diet_goal(query: str) -> str:
    q = clean_query(query)
    if any(p in q for p in ["lose weight", "loss weight", "cut weight", "reduce weight", "lose fat", "cut fat"]):
        return "weight loss"
    if any(p in q for p in ["gain weight", "weight gain", "build muscle", "muscle gain", "mass gain"]):
        return "weight gain"
    if "loss" in q:
        return "weight loss"
    if "gain" in q or "bulk" in q:
        return "weight gain"
    return "general"

@lru_cache(maxsize=128)
def extract_regional_preference(query: str) -> str:
    q = clean_query(query)
    for region in REGION_KEYWORDS:
        if region in q:
            return region.title()
    return "Indian"

@lru_cache(maxsize=128)
def extract_disease_condition(query: str) -> Optional[str]:
    q = clean_query(query)
    for disease, keywords in DISEASE_KEYWORDS.items():
        if any(k in q for k in keywords):
            return disease
    return None

@lru_cache(maxsize=128)
def is_follow_up_query(query: str) -> bool:
    q = clean_query(query)
    followup_phrases = ["same but", "make it", "change to", "instead", "now", "also", "can you", "but make it"]
    return any(p in q for p in followup_phrases)

def extract_all_metadata(query: str) -> dict:
    return {
        "dietary_type": extract_diet_preference(query),
        "goal": extract_diet_goal(query),
        "region": extract_regional_preference(query),
        "disease": extract_disease_condition(query),
        "formatting": is_formatting_request(query),
        "wants_table": contains_table_request(query),
        "is_follow_up": is_follow_up_query(query),
        "is_greeting": is_greeting(query),
        "is_generic": is_generic_query(query)
    }

