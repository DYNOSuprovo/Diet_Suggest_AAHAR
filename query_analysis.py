# query_analysis.py
import string
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GREETINGS = ["hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam", "good morning", "good afternoon", "good evening"]
TASK_KEYWORDS = ["diet", "plan", "weight", "gain", "loss", "table", "format", "chart", "show", "give", "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian", "south", "north", "east", "west", "india", "bengali", "punjabi", "maharashtrian", "gujarati", "tamil", "kannada", "telugu", "malayalam", "kanyakumari", "odisha", "oriya", "bhubaneswar", "cuttack", "angul"]
FORMATTING_KEYWORDS = ["table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate"]

def is_greeting(query: str) -> bool:
    """Checks if the query is a pure greeting."""
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    contains_task_keyword = any(keyword in cleaned_query for keyword in TASK_KEYWORDS)
    is_short_greeting = cleaned_query in GREETINGS and len(cleaned_query.split()) <= 3
    return is_short_greeting and not contains_task_keyword

def is_formatting_request(query: str) -> bool:
    """Checks if the query is primarily a formatting request."""
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    words = cleaned_query.split()

    if not any(keyword in cleaned_query for keyword in FORMATTING_KEYWORDS):
        return False

    if len(words) <= 5:
        non_formatting_words = 0
        for word in words:
            if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you"]:
                non_formatting_words += 1
        if non_formatting_words <= 1:
            logging.info(f"Query '{query}' identified as formatting request (short, specific keywords).")
            return True

    only_formatting_and_fillers = True
    for word in words:
        if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "the", "for", "my", "me"]:
            only_formatting_and_fillers = False
            break
    if only_formatting_and_fillers:
        logging.info(f"Query '{query}' identified as formatting request (only formatting/fillers).")
        return True
    return False

def extract_diet_preference(query: str) -> str:
    """Extracts dietary preference from the query."""
    query_lower = query.lower()
    if "non-veg" in query_lower or "non veg" in query_lower or "nonvegetarian" in query_lower: return "non-vegetarian"
    if "vegan" in query_lower: return "vegan"
    if "veg" in query_lower or "vegetarian" in query_lower: return "vegetarian"
    return "any"

def extract_diet_goal(query: str) -> str:
    """
    Extracts diet goal from the query, prioritizing weight loss if explicitly requested,
    even if current state implies weight gain.
    """
    query_lower = query.lower()

    # Prioritize clear weight loss intentions, including phrases like "cut weight/fat"
    if "lose weight" in query_lower or "loss weight" in query_lower or \
       "cut weight" in query_lower or "reduce weight" in query_lower or \
       "lose fat" in query_lower or "cut fat" in query_lower:
        return "weight loss"

    # Then check for clear weight gain intentions
    if "gain weight" in query_lower or "weight gain" in query_lower:
        return "weight gain"

    # Handle general "loss" or "gain" keywords as a fallback,
    # ensuring "loss" is still preferred if the above didn't catch a specific phrase
    if "loss" in query_lower:
        return "weight loss"
    if "gain" in query_lower:
        # Only return "weight gain" if no strong "loss" indicators were found
        return "weight gain"

    return "diet" # Default if no specific goal is found

def extract_regional_preference(query: str) -> str:
    """Extracts regional preference from the query."""
    query_lower = query.lower()
    match = re.search(r"(south|north|west|east) india(?:n)?|bengali|punjabi|maharashtrian|gujarati|tamil|kannada|telugu|malayalam|kanyakumari|odisha|oriya|bhubaneswar|cuttack|angul", query_lower)
    if match: return " ".join([word.capitalize() for word in match.group(0).replace('indian', 'Indian').split()])
    return "Indian"

def contains_table_request(query: str) -> bool:
    """Checks if the query contains a request for table format."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])# query_analysis.py
import string
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GREETINGS = ["hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam", "good morning", "good afternoon", "good evening"]
TASK_KEYWORDS = ["diet", "plan", "weight", "gain", "loss", "table", "format", "chart", "show", "give", "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian", "south", "north", "east", "west", "india", "bengali", "punjabi", "maharashtrian", "gujarati", "tamil", "kannada", "telugu", "malayalam", "kanyakumari", "odisha", "oriya", "bhubaneswar", "cuttack", "angul"]
FORMATTING_KEYWORDS = ["table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate"]

def is_greeting(query: str) -> bool:
    """Checks if the query is a pure greeting."""
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    contains_task_keyword = any(keyword in cleaned_query for keyword in TASK_KEYWORDS)
    is_short_greeting = cleaned_query in GREETINGS and len(cleaned_query.split()) <= 3
    return is_short_greeting and not contains_task_keyword

def is_formatting_request(query: str) -> bool:
    """Checks if the query is primarily a formatting request."""
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    words = cleaned_query.split()

    if not any(keyword in cleaned_query for keyword in FORMATTING_KEYWORDS):
        return False

    if len(words) <= 5:
        non_formatting_words = 0
        for word in words:
            if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you"]:
                non_formatting_words += 1
        if non_formatting_words <= 1:
            logging.info(f"Query '{query}' identified as formatting request (short, specific keywords).")
            return True

    only_formatting_and_fillers = True
    for word in words:
        if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "the", "for", "my", "me"]:
            only_formatting_and_fillers = False
            break
    if only_formatting_and_fillers:
        logging.info(f"Query '{query}' identified as formatting request (only formatting/fillers).")
        return True
    return False

def extract_diet_preference(query: str) -> str:
    """Extracts dietary preference from the query."""
    query_lower = query.lower()
    if "non-veg" in query_lower or "non veg" in query_lower or "nonvegetarian" in query_lower: return "non-vegetarian"
    if "vegan" in query_lower: return "vegan"
    if "veg" in query_lower or "vegetarian" in query_lower: return "vegetarian"
    return "any"

def extract_diet_goal(query: str) -> str:
    """
    Extracts diet goal from the query, prioritizing weight loss if explicitly requested,
    even if current state implies weight gain.
    """
    query_lower = query.lower()

    # Prioritize clear weight loss intentions, including phrases like "cut weight/fat"
    if "lose weight" in query_lower or "loss weight" in query_lower or \
       "cut weight" in query_lower or "reduce weight" in query_lower or \
       "lose fat" in query_lower or "cut fat" in query_lower:
        return "weight loss"

    # Then check for clear weight gain intentions
    if "gain weight" in query_lower or "weight gain" in query_lower:
        return "weight gain"

    # Handle general "loss" or "gain" keywords as a fallback,
    # ensuring "loss" is still preferred if the above didn't catch a specific phrase
    if "loss" in query_lower:
        return "weight loss"
    if "gain" in query_lower:
        # Only return "weight gain" if no strong "loss" indicators were found
        return "weight gain"

    return "diet" # Default if no specific goal is found

def extract_regional_preference(query: str) -> str:
    """Extracts regional preference from the query."""
    query_lower = query.lower()
    match = re.search(r"(south|north|west|east) india(?:n)?|bengali|punjabi|maharashtrian|gujarati|tamil|kannada|telugu|malayalam|kanyakumari|odisha|oriya|bhubaneswar|cuttack|angul", query_lower)
    if match: return " ".join([word.capitalize() for word in match.group(0).replace('indian', 'Indian').split()])
    return "Indian"

def contains_table_request(query: str) -> bool:
    """Checks if the query contains a request for table format."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])