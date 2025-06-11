# query_analysis.py
import string
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Core keyword sets
GREETINGS = {"hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam", "good morning", "good afternoon", "good evening"}
TASK_KEYWORDS = {
    "diet", "plan", "weight", "gain", "loss", "table", "format", "chart", "show", "give", 
    "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian",
    "south", "north", "east", "west", "india", "bengali", "punjabi", "maharashtrian", 
    "gujarati", "tamil", "kannada", "telugu", "malayalam", "kanyakumari", 
    "odisha", "oriya", "bhubaneswar", "cuttack"
}
FORMATTING_KEYWORDS = {"table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate"}
FILLER_WORDS = {"in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "the", "for", "my"}

def clean_query(query: str) -> str:
    """Strips punctuation and lowercases the query."""
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

def is_greeting(query: str) -> bool:
    """Checks if the query is a pure greeting."""
    if not query:
        return False
    cleaned = clean_query(query)
    words = cleaned.split()
    contains_task = any(keyword in cleaned for keyword in TASK_KEYWORDS)
    return cleaned in GREETINGS and len(words) <= 3 and not contains_task

def is_formatting_request(query: str) -> bool:
    """Checks if the query is primarily a formatting request."""
    if not query:
        return False
    cleaned = clean_query(query)
    words = cleaned.split()

    if not any(k in cleaned for k in FORMATTING_KEYWORDS):
        return False

    if len(words) <= 5:
        non_formatting = sum(1 for w in words if w not in FORMATTING_KEYWORDS and w not in FILLER_WORDS)
        if non_formatting <= 1:
            logging.info(f"Query '{query}' identified as formatting request (short, specific keywords).")
            return True

    if all(w in FORMATTING_KEYWORDS or w in FILLER_WORDS for w in words):
        logging.info(f"Query '{query}' identified as formatting request (only formatting/fillers).")
        return True

    return False

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

def extract_diet_goal(query: str) -> str:
    """Extracts diet goal from the query."""
    q = query.lower()
    if any(p in q for p in ["lose weight", "loss weight", "cut weight", "reduce weight", "lose fat", "cut fat"]):
        return "weight loss"
    if "gain weight" in q or "weight gain" in q:
        return "weight gain"
    if "loss" in q:
        return "weight loss"
    if "gain" in q:
        return "weight gain"
    return "diet"

def extract_regional_preference(query: str) -> str:
    """Extracts regional preference from the query."""
    q = query.lower()
    match = re.search(
        r"\b((south|north|west|east)\s+indian|bengali|punjabi|maharashtrian|gujarati|tamil|kannada|telugu|malayalam|kanyakumari|odisha|oriya|bhubaneswar|cuttack|angul)\b",
        q
    )
    if match:
        return " ".join([word.capitalize() for word in match.group(0).split()])
    return "Indian"

def contains_table_request(query: str) -> bool:
    """Checks if the query contains a request for tabular format."""
    q = query.lower()
    return any(k in q for k in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])
