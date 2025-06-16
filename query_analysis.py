import string
import re
import logging
from functools import lru_cache
from typing import Optional, Dict, Any

# Configure logging for better insights
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Keyword Sets ---
GREETINGS = {
    "hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam",
    "good morning", "good afternoon", "good evening", "heelo",
    "how are you", "you there", "are you there", "sup", "howdy",
    "greetings", "aloha", "hola", "ciao", "wassup", "what's up",
    "hello there", "can you hear me", "do you understand"
}

TASK_KEYWORDS = {
    "diet", "plan", "meal", "food", "eat", "what to eat", "suggestion", "recommend",
    "weight", "gain", "loss", "muscle", "mass", "healthy", "unhealthy",
    "calorie", "protein", "fiber", "carb", "fat", "nutrition", "nutrients",
    "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian",
    "south", "north", "east", "west", "india", "indian",
    "bengali", "punjabi", "maharashtrian", "gujarati", "tamil", "kannada",
    "telugu", "malayalam", "kanyakumari", "odisha", "oriya", "bhubaneswar",
    "cuttack", "angul", "rajasthan", "jaisalmir",
    "diabetes", "bp", "sugar", "pressure", "cholesterol", "diabetic", "hypertension", "heart disease",
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", # Cities for regional context
    "kidney", "renal", "liver", "hepatic", "thyroid", "hypothyroid", "hyperthyroid", "pcos", "pcod", # Expanded diseases
    "recipes", "recipe", "dishes", "dish", "breakfast", "lunch", "dinner", "snack" # Added meal times as task keywords
}

# GENERIC_INTENT_PHRASES: Strictly for non-diet, meta-questions or pure exclamations.
GENERIC_INTENT_PHRASES = {
    "who are you", "who created you", "what is this", "what are you", "where am i",
    "what do you do", "what can you do", "lol", "lmao", "haha", "what", "tell me about yourself",
    "explain yourself", "can you help", "how are things", "anything else",
    "help", "help me", "what can you help with", "what services", "what functions",
    "how do you work", "how does this work", "explain this", "what's this for",
    "hmm", "ok", "okay", "cool", "nice", "wow", "interesting", "really",
    "seriously", "no way", "awesome", "great", "something", "anything", "whatever",
    "i don't know", "not sure", "maybe", "perhaps", "could be", "might be",
    "sort of", "kind of", "what's new", "what's happening", "any updates",
    "tell me something", "surprise me", "entertain me", "test", "testing",
    "check", "checking", "are you working", "ty", "thx", "appreciate",
    "sorry", "apologize", "my bad", "oops", "bye", "goodbye", "see you", "later",
    "farewell", "ciao", "just kidding", "just joking"
}

# Pattern-based generic detection (for short, non-phrase matches like "?", "ok")
GENERIC_PATTERNS = [
    r'^(what|how|why|when|where|who)\s*\?*$',
    r'^(yes|no|yeah|nah|yep|nope|sure|fine|ok|okay)\s*$',
    r'^(thanks|thank you)\s*$',
    r'^(sorry|my bad)\s*$',
    r'^(bye|goodbye)\s*$',
    r'^\w{1,2}$', # Very short words like "uh", "um", "ah"
    r'^[?!.]{1,3}$', # Just punctuation
    r'^(hm+|um+|uh+|ah+|oh+)\s*$'
]

FORMATTING_KEYWORDS = {
    "table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate",
    "structured", "organized", "paragraph", "in paragraph", "in table", "line by line",
    "grid", "column", "row", "spreadsheet", "csv", # Table-related
    "briefly", "summary", "summarize", "concise", "short", "long", "detailed", "full" # Length/style related
}

# MODIFICATION_KEYWORDS: Explicitly for modifying a *plan's content*
MODIFICATION_KEYWORDS = {
    "remove", "delete", "exclude", "omit", "without", "no", "minus",
    "add", "include", "with", "plus",
    "change", "modify", "update", "alter", "different", "another", "alternative", "variation",
    "not do", "don't do", "skip" # Added more natural language for exclusions
}

FILLER_WORDS = {
    "in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you",
    "the", "for", "my", "to", "with", "want", "now", "like", "need", "i", "am", "is",
    "how", "what", "do", "you", "just", "and", "or", "but", "an", "on", "at", "from",
    "about", "some", "any", "this", "that", "get", "make", "create", "build",
    "generate", "provide", "help", "assist", "a bit", "a little", "of course", "generally" # Added "generally"
}

REGION_KEYWORDS = [
    "south indian", "north indian", "east indian", "west indian", "bengali", "punjabi",
    "maharashtrian", "gujarati", "tamil", "kannada", "telugu", "malayalam", "odisha",
    "oriya", "kanyakumari", "bhubaneswar", "cuttack", "angul", "rajasthan", "jaisalmir",
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad",
    "kerala", "andhra", "telangana", "uttar pradesh", "bihar", "west bengal", "goa",
    "haryana", "himachal", "jammu", "kashmir", "karnataka", "madhya pradesh", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "punjab", "sikkim", "tripura", "uttarakhand"
]

DISEASE_KEYWORDS = {
    "diabetes": ["diabetes", "sugar", "diabetic", "glucose", "insulin", "blood sugar"],
    "blood pressure": ["bp", "blood pressure", "pressure", "hypertension", "high bp", "low bp"],
    "cholesterol": ["cholesterol", "lipid", "heart disease", "high cholesterol", "hdl", "ldl", "triglycerides", "cardiac"],
    "kidney": ["kidney", "renal", "nephritis", "kidney stones"],
    "liver": ["liver", "hepatic", "fatty liver", "jaundice"],
    "thyroid": ["thyroid", "hypothyroid", "hyperthyroid", "thyroxin"],
    "pcos": ["pcos", "pcod", "polycystic ovary syndrome"],
    "digestion": ["digestion", "digestive", "gut health", "irritable bowel", "constipation", "diarrhea"],
    "anemia": ["anemia", "iron deficiency", "low iron"],
    "arthritis": ["arthritis", "joint pain", "gout"]
}


# --- Utility Functions ---

def clean_query(query: str) -> str:
    """
    Cleans the input query: removes punctuation, converts to lowercase,
    and normalizes whitespace.
    """
    if not query:
        return ""
    cleaned = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip() # Normalize multiple spaces
    return cleaned

@lru_cache(maxsize=128)
def is_greeting(query: str) -> bool:
    """
    Checks if the query is a simple greeting.
    A greeting is typically short and does not contain clear task-oriented keywords.
    """
    cleaned = clean_query(query)
    words = cleaned.split()

    # Direct match for efficiency
    if cleaned in GREETINGS:
        logging.info(f"✨ Detected exact greeting: '{query}'")
        return True

    # Check if a significant portion are greeting words and no task words
    greeting_words_in_query = [w for w in words if w in GREETINGS]
    if greeting_words_in_query and not any(k in cleaned for k in TASK_KEYWORDS):
        if len(words) <= 5: # Short greetings
            logging.info(f"✨ Detected indirect greeting (short & non-task): '{query}'")
            return True
        non_filler_words = [w for w in words if w not in FILLER_WORDS]
        if len(non_filler_words) > 0 and len(greeting_words_in_query) / len(non_filler_words) >= 0.7:
             logging.info(f"✨ Detected indirect greeting (high greeting word density): '{query}'")
             return True

    return False

@lru_cache(maxsize=128)
def is_generic_query(query: str) -> bool:
    """
    Checks if the query is truly generic, non-task-oriented, or a meta-question.
    This function returns True only if the query cannot be classified as a greeting,
    a task-oriented request, or a primary formatting/modification request.
    """
    if not query:
        return True # Empty query is generic

    q = clean_query(query)
    words = q.split()

    # Rule 1: Matches known generic phrases
    if q in GENERIC_INTENT_PHRASES:
        logging.info(f"❓ Detected generic (phrase match): '{query}'")
        return True

    # Rule 2: Pattern matching (e.g., single question words, simple responses)
    for pattern in GENERIC_PATTERNS:
        if re.match(pattern, q):
            logging.info(f"🔍 Generic (pattern match): {query}")
            return True

    # Check for any task, formatting, or modification keyword
    has_specific_keywords = any(k in q for k in TASK_KEYWORDS) or \
                            any(k in q for k in FORMATTING_KEYWORDS) or \
                            any(k in q for k in MODIFICATION_KEYWORDS)

    # If it contains any specific keyword, it's NOT generic.
    if has_specific_keywords:
        logging.info(f"🎯 Not generic (contains specific keyword): '{query}'")
        return False

    # Rule 3: Very short queries (1-4 words) that are NOT greetings and contain NO specific keywords.
    if 1 <= len(words) <= 4:
        if not is_greeting(query):
            logging.info(f"❓ Detected generic (short, no specific keywords, not greeting): '{query}'")
            return True

    # Rule 4: Queries where all non-filler words are absent or very few, and no specific keywords.
    non_filler_words = [w for w in words if w not in FILLER_WORDS]
    if len(non_filler_words) == 0: # All words are fillers
        if not is_greeting(query):
            logging.info(f"❓ Detected generic (all fillers, no specific keywords, not greeting): '{query}'")
            return True

    return False

@lru_cache(maxsize=128)
def is_formatting_request(query: str) -> bool:
    """
    Checks if the query primarily asks for output formatting OR is a direct modification command.
    This function specifically targets requests to re-format or subtly alter the structure/content
    of a previous response, NOT a request for a new diet plan with a constraint.
    """
    if not query:
        return False
    cleaned = clean_query(query)
    words = cleaned.split()

    has_format_keyword = any(k in cleaned for k in FORMATTING_KEYWORDS)
    has_modification_keyword = any(k in cleaned for k in MODIFICATION_KEYWORDS)

    # If it has *only* formatting or modification keywords (and fillers) and is short, it's a formatting request
    if (has_format_keyword or has_modification_keyword) and \
       not any(k in cleaned for k in TASK_KEYWORDS) and \
       len(words) <= 6: # Keep this short to avoid catching full task queries
        non_filler_and_specific_words = [w for w in words if w not in FILLER_WORDS and (w in FORMATTING_KEYWORDS or w in MODIFICATION_KEYWORDS)]
        if len(non_filler_and_specific_words) > 0 and len(non_filler_and_specific_words) == len([w for w in words if w not in FILLER_WORDS]):
            logging.info(f"⚙️ Formatting/Modification (short, no task, specific words only): '{query}'")
            return True
        
        # Catch explicit formatting/modification phrases like "remove breakfast" if no other strong task keywords
        if "remove breakfast" in cleaned or "no breakfast" in cleaned:
             if not any(k in cleaned for k in TASK_KEYWORDS if k not in ["breakfast", "meal"]): # "breakfast" can be a task keyword, but 'remove breakfast' is a modification here
                 logging.info(f"⚙️ Formatting/Modification (explicit exclusion): '{query}'")
                 return True


    # If it contains formatting keywords and is a follow-up, it could be a reformat request
    # This logic overlaps with is_follow_up_query, but helps specifically identify the *type* of follow-up.
    # Consider "give in table format" - this is primarily formatting.
    if has_format_keyword and (len(words) <= 5 or all(w in FORMATTING_KEYWORDS or w in FILLER_WORDS for w in words)):
        if not any(k in cleaned for k in TASK_KEYWORDS): # Must not also be a task
            logging.info(f"⚙️ Formatting request (pure formatting, possibly follow-up): '{query}'")
            return True

    return False


@lru_cache(maxsize=128)
def contains_table_request(query: str) -> bool:
    """Detects if the query explicitly asks for a table format."""
    q = clean_query(query)
    table_indicators = [
        "table", "tabular", "chart", "in a table", "in table format",
        "as a table", "spreadsheet", "grid format", "column", "row"
    ]
    return any(indicator in q for indicator in table_indicators)

@lru_cache(maxsize=128)
def contains_paragraph_request(query: str) -> bool:
    """Detects if the query explicitly asks for a paragraph format."""
    q = clean_query(query)
    return any(k in q for k in ["paragraph", "in paragraph form", "in paragraph", "prose", "text form"])

@lru_cache(maxsize=128)
def contains_modification_request(query: str) -> bool:
    """Detects if the query explicitly asks for a modification (remove/add)."""
    q = clean_query(query)
    return any(k in q for k in MODIFICATION_KEYWORDS)


@lru_cache(maxsize=128)
def extract_diet_preference(query: str) -> str:
    """Extracts dietary preference (non-vegetarian, vegan, vegetarian, any)."""
    q = clean_query(query)
    if any(x in q for x in ["non-veg", "non veg", "nonvegetarian", "meat", "chicken", "fish", "egg"]):
        return "non-vegetarian"
    if "vegan" in q:
        return "vegan"
    if any(x in q for x in ["veg", "vegetarian", "plant based"]):
        return "vegetarian"
    return "any"

@lru_cache(maxsize=128)
def extract_diet_goal(query: str) -> str:
    """Extracts diet goal (weight loss, weight gain, general)."""
    q = clean_query(query)
    weight_loss_patterns = [
        "lose weight", "loss weight", "cut weight", "reduce weight",
        "lose fat", "cut fat", "slim down", "get lean", "shed pounds", "slimming", "lean"
    ]
    weight_gain_patterns = [
        "gain weight", "weight gain", "build muscle", "muscle gain",
        "mass gain", "bulk up", "get bigger", "add mass", "bulk", "bulking"
    ]

    if any(pattern in q for pattern in weight_loss_patterns):
        return "weight loss"
    if any(pattern in q for pattern in weight_gain_patterns):
        return "weight gain"

    # Specific checks for "loss" or "gain" not in other contexts
    if "loss" in q and not any(kw in q for kw in ["data", "money", "time", "sleep"]):
        return "weight loss"
    if "gain" in q and not any(kw in q for kw in ["knowledge", "experience", "insight"]):
        return "weight gain"

    return "general"

@lru_cache(maxsize=128)
def extract_regional_preference(query: str) -> str:
    """Extracts regional food preference from the query."""
    q = clean_query(query)
    for region in REGION_KEYWORDS:
        if region in q:
            # Handle multi-word regions correctly: "South Indian"
            return " ".join(word.capitalize() for word in region.split())
    
    if "indian" in q:
        return "Indian" # General Indian if no specific region
    return "Indian" # Default if no specific region or "indian" is mentioned

@lru_cache(maxsize=128)
def extract_disease_condition(query: str) -> Optional[str]:
    """Extracts specific disease conditions from the query."""
    q = clean_query(query)
    for disease, keywords in DISEASE_KEYWORDS.items():
        if any(k in q for k in keywords):
            return disease
    return None

@lru_cache(maxsize=128)
def is_follow_up_query(query: str) -> bool:
    """
    Checks if the query is likely a follow-up to a previous interaction.
    Looks for common follow-up phrases or very short queries that seem to modify context.
    Now also considers explicit modification keywords like "remove".
    """
    q = clean_query(query)
    followup_phrases = {
        "same but", "make it", "change to", "instead", "now", "also",
        "can you", "could you", "would you", "will you",
        "but make it", "what about", "and also", "next", "then", "continue",
        "how about", "modify", "update", "again", "for me", "just",
        "different", "another", "alternative", "variation", "reformat", "re-format",
        # Explicit modification phrases
        "this", "that", "it", # Referring to previous content
        "remove this", "delete that", "exclude", "omit", "without", "no breakfast",
        "add this", "include that", "skip" # Added "skip"
    }

    # Check for direct follow-up phrases or explicit modification keywords
    if any(p in q for p in followup_phrases) or any(k in q for k in MODIFICATION_KEYWORDS):
        logging.info(f"🔄 Detected follow-up (phrase or modification keyword match): '{query}'")
        return True

    # Check for very short queries with some task/formatting/modification keywords, implying modification
    words = q.split()
    non_filler_words = [w for w in words if w not in FILLER_WORDS]
    if len(words) <= 5 and len(non_filler_words) > 0 and len(non_filler_words) <= 3:
        if any(k in non_filler_words for k in TASK_KEYWORDS) or \
           any(k in non_filler_words for k in FORMATTING_KEYWORDS) or \
           any(k in non_filler_words for k in MODIFICATION_KEYWORDS):
             logging.info(f"🔄 Detected follow-up (short, context-modifying): '{query}'")
             return True

    return False

def get_query_confidence_score(query: str) -> float:
    """Calculate confidence score for query classification based on task-related content."""
    if not query:
        return 0.0

    q = clean_query(query)
    words = q.split()

    if len(words) == 0:
        return 0.0

    # Calculate task-specific word ratio
    task_words = sum(1 for word in words if word in TASK_KEYWORDS)
    task_ratio = task_words / len(words)

    # Calculate meaningful content ratio (non-filler words)
    meaningful_words = sum(1 for word in words if word not in FILLER_WORDS)
    meaningful_ratio = meaningful_words / len(words) if len(words) > 0 else 0

    # Combine ratios for confidence score (can be adjusted based on desired weighting)
    confidence = (task_ratio * 0.7) + (meaningful_ratio * 0.3)

    return min(confidence, 1.0)


def extract_all_metadata(query: str) -> Dict[str, Any]:
    """
    Extracts all relevant metadata from the query, determining its primary intent.
    The hierarchy for primary intent is: Greeting -> Formatting/Modification -> Generic -> Task.
    """
    logging.info(f"Starting analysis for query: '{query}'")

    cleaned_q = clean_query(query)
    confidence = get_query_confidence_score(query)

    # Determine primary intent type in a hierarchical manner
    primary_intent_type = "generic" # Default to generic as the lowest priority

    if is_greeting(cleaned_q):
        primary_intent_type = "greeting"
    # Check for a formatting/modification request that isn't also a strong task keyword query
    elif is_formatting_request(cleaned_q): # The logic for this function is now more precise
        primary_intent_type = "formatting"
    elif is_generic_query(cleaned_q):
        primary_intent_type = "generic" # This will catch remaining generic queries
    elif any(k in cleaned_q for k in TASK_KEYWORDS) or contains_modification_request(cleaned_q): # Modified to include modification keywords in task
        primary_intent_type = "task" # Catch any query with task keywords or modification keywords (like "remove breakfast")


    # Extract specific details if it's potentially a task or formatting/modification request
    dietary_type = "any"
    goal = "general"
    region = "Indian"
    disease = None
    wants_table = False
    wants_paragraph = False
    is_follow_up_flag = False # Renamed to avoid conflict with function name
    
    # Extract these only if it's not a pure greeting or pure generic query (i.e., it has some intent)
    if primary_intent_type not in ["greeting", "generic"]:
        dietary_type = extract_diet_preference(cleaned_q)
        goal = extract_diet_goal(cleaned_q)
        region = extract_regional_preference(cleaned_q)
        disease = extract_disease_condition(cleaned_q)
        is_follow_up_flag = is_follow_up_query(cleaned_q) # This will now be true for "remove breakfast"

    # Formatting flags should always be checked if the query has formatting keywords,
    # as they indicate a desired output style regardless of primary intent.
    wants_table = contains_table_request(cleaned_q)
    wants_paragraph = contains_paragraph_request(cleaned_q)

    # Final adjustment for `is_follow_up_flag`:
    # If it's a modification request (like "remove breakfast"), it's definitely a follow-up.
    # This ensures it's treated as a follow-up for the LLM to process as a constraint.
    if contains_modification_request(cleaned_q):
        is_follow_up_flag = True


    metadata = {
        "primary_intent_type": primary_intent_type,
        "dietary_type": dietary_type,
        "goal": goal,
        "region": region,
        "disease": disease,
        "wants_table": wants_table,
        "wants_paragraph": wants_paragraph,
        "is_follow_up": is_follow_up_flag, # Use the flag
        "confidence_score": confidence,
        "query_length": len(query.split()) if query else 0,
        "has_task_keywords": any(k in cleaned_q for k in TASK_KEYWORDS), # Boolean for task keywords
        "has_formatting_keywords": any(k in cleaned_q for k in FORMATTING_KEYWORDS), # Boolean for formatting keywords
        "has_modification_keywords": any(k in cleaned_q for k in MODIFICATION_KEYWORDS) # Boolean for modification keywords
    }
    logging.info(f"Final extracted metadata: {metadata}")
    return metadata

# --- Test Cases (for demonstration and understanding) ---
if __name__ == "__main__":
    test_queries = [
        # Greeting Tests
        "hi", "hello there", "how are you", "what's up", "namaste",

        # Generic Tests (should be classified as 'generic')
        "what", "who are you", "tell me about yourself", "can you help", "lol", "?", "ok",
        "help me", "how do you work", "thanks", "bye", "test", "what is this", "anything",

        # Task-Oriented Tests (should be classified as 'task')
        "diet plan for weight loss", "show me vegetarian diet", "give a Puri based diet chart for a 19 year old boy",
        "suggest diet for diabetes in north indian style", "food for weight loss",
        "can you give me a plan for high bp", "I need a diet for thyroid",
        "plan for plant based diet", "something for healthy heart", "give me some suggestions",
        "diet for me", "what can I eat", "recipes for dinner",

        # Formatting/Modification Tests (should be 'formatting' or 'task' with formatting/follow-up)
        "show a diet plan in table format", # Task + Formatting (primary_intent_type: task, wants_table: True, is_follow_up: True)
        "give in table format",             # Formatting + Follow-up (primary_intent_type: formatting, wants_table: True, is_follow_up: True)
        "in paragraph form only",           # Formatting + Follow-up (primary_intent_type: formatting, wants_paragraph: True, is_follow_up: True)
        "just table",                       # Formatting + Follow-up (primary_intent_type: formatting, wants_table: True, is_follow_up: True)
        "remove breakfast",                 # NEW: Modification + Follow-up (primary_intent_type: task, is_follow_up: True, has_modification_keywords: True, dietary_type/goal/region/disease from context/memory)
        "add more protein",                 # NEW: Modification + Follow-up (primary_intent_type: task, is_follow_up: True, has_modification_keywords: True)
        "make it vegan",                    # Follow-up + Preferences (primary_intent_type: task, is_follow_up: True)
        "same but make it non-veg",         # Follow-up + Preferences (primary_intent_type: task, is_follow_up: True)
        "can you make it shorter",          # Formatting + Follow-up (primary_intent_type: formatting, is_follow_up: True)
        "no dinner"                         # NEW: Modification + Follow-up (primary_intent_type: task, is_follow_up: True, has_modification_keywords: True)
    ]

    print("--- Consolidated Query Analysis Test Results ---")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        metadata = extract_all_metadata(q)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    print("\n--- End of Test Results ---")
