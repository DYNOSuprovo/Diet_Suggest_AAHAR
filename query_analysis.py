import string
import re
import logging
from functools import lru_cache
from typing import Optional, Dict, Any

# Configure logging for better insights
# Ensure logging is configured to output to standard error or a file where it can be monitored.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Keyword Sets ---
# These sets are expanded to improve the accuracy of query classification.
GREETINGS = {
    "hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam",
    "good morning", "good afternoon", "good evening", "heelo", # Added 'heelo' from user logs
    "how are you", "you there", "are you there", "sup", "howdy",
    "greetings", "aloha", "hola", "ciao", "wassup", "what's up" # Combined from first snippet
}

TASK_KEYWORDS = {
    "diet", "plan", "meal", "food", "eat", "what to eat", "suggestion", "recommend",
    "weight", "gain", "loss", "muscle", "mass", "healthy", "unhealthy",
    "calorie", "protein", "fiber", "carb", "fat", "nutrition", "nutrients",
    "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian",
    "south", "north", "east", "west", "india", "indian", # Explicitly added 'indian'
    "bengali", "punjabi", "maharashtrian", "gujarati", "tamil", "kannada",
    "telugu", "malayalam", "kanyakumari", "odisha", "oriya", "bhubaneswar",
    "cuttack", "angul", "rajasthan", "jaisalmir", # Added Jaisalmir and Rajasthan for regional context
    "diabetes", "bp", "sugar", "pressure", "cholesterol", "diabetic", "hypertension", "heart disease" # Expanded disease terms
    # No significant unique TASK_KEYWORDS in first snippet to add here.
}

# GENERIC_INTENT_PHRASES: More strictly for non-diet, meta-questions or pure exclamations.
# These queries typically don't require a diet suggestion.
GENERIC_INTENT_PHRASES = {
    "who are you", "who created you", "what is this", "what are you", "where am i",
    "what do you do", "what can you do", "lol", "lmao", "haha", "what", "tell me about yourself",
    "explain yourself", "can you help", "how are things", "what's up", "anything else",
    "help", "help me", "what can you help with", "what services", "what functions", # From first snippet
    "how do you work", "how does this work", "explain this", "what's this for", # From first snippet
    "hmm", "ok", "okay", "cool", "nice", "wow", "interesting", "really", # From first snippet
    "seriously", "no way", "awesome", "great", "something", "anything", "whatever", # From first snippet
    "i don't know", "not sure", "maybe", "perhaps", "could be", "might be", # From first snippet
    "sort of", "kind of", "how are you", "how's it going", "what's new", # From first snippet
    "what's happening", "any updates", "tell me something", "surprise me", "entertain me", # From first snippet
    "test", "testing", "check", "checking", "are you there", "are you working", # From first snippet
    "hello there", "can you hear me", "do you understand" # From first snippet
}

# Pattern-based generic detection from the first snippet (can be useful for short, non-phrase matches)
GENERIC_PATTERNS = [
    r'^(what|how|why|when|where|who)\s*\?*$',  # Single question words
    r'^(yes|no|yeah|nah|yep|nope|sure|fine|ok|okay)\s*$',  # Simple responses
    r'^(thanks|thank you|ty|thx|appreciate)\s*$',  # Gratitude
    r'^(sorry|apologize|my bad|oops)\s*$',  # Apologies
    r'^(bye|goodbye|see you|later|farewell|ciao)\s*$',  # Farewells
    r'^\w{1,2}$',  # Very short responses (1-2 chars)
    r'^[a-z]{1,3}$',  # Short letter combinations
    r'^\d+$',  # Just numbers
    r'^[?!.]{1,3}$',  # Just punctuation
    r'^(hm+|um+|uh+|ah+|oh+)\s*$'  # Thinking sounds
]


FORMATTING_KEYWORDS = {
    "table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate",
    "structured", "organized", "paragraph", "in paragraph", "in table", "line by line", # Added paragraph and line by line
    "grid", "column", "row", "spreadsheet", "csv" # From first snippet
}

# FILLER_WORDS: Used to filter out less informative words in short queries.
FILLER_WORDS = {
    "in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you",
    "the", "for", "my", "to", "with", "want", "now", "like", "need", "i", "am", "is",
    "how", "what", "do", "you", "just", "and", "or", "but", "an", "on", "at", "from",
    "about", "some", "any", "this", "that", "get", "make", "create", "build", # From first snippet
    "generate", "provide", "help", "assist" # From first snippet
}

REGION_KEYWORDS = [
    "south indian", "north indian", "east indian", "west indian", "bengali", "punjabi",
    "maharashtrian", "gujarati", "tamil", "kannada", "telugu", "malayalam", "odisha",
    "oriya", "kanyakumari", "bhubaneswar", "cuttack", "angul", "rajasthan", "jaisalmir",
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad" # From first snippet (cities)
]

DISEASE_KEYWORDS = {
    "diabetes": ["diabetes", "sugar", "diabetic", "glucose", "insulin"], # Combined with first snippet keywords
    "blood pressure": ["bp", "blood pressure", "pressure", "hypertension", "high bp", "low bp"], # Combined
    "cholesterol": ["cholesterol", "lipid", "heart disease", "high cholesterol", "hdl", "ldl", "triglycerides"], # Combined
    "heart disease": ["heart", "cardiac", "cardiovascular"], # From first snippet, ensure no conflict
    "kidney": ["kidney", "renal", "nephritis"], # From first snippet
    "liver": ["liver", "hepatic", "fatty liver"], # From first snippet
    "thyroid": ["thyroid", "hypothyroid", "hyperthyroid"], # Added common disease
    "pcos": ["pcos", "pcod"] # Added common disease
}

# Conversation context patterns from first snippet
CONVERSATION_STARTERS = {
    "i want", "i need", "can you", "could you", "would you", "will you",
    "how to", "how do i", "what should", "should i", "tell me",
    "explain", "describe", "define", "what is", "what are"
}


# --- Utility Functions ---

def clean_query(query: str) -> str:
    """
    Cleans the input query:
    1. Removes punctuation. (Second snippet's aggressive punctuation removal is generally better)
    2. Converts to lowercase.
    3. Strips leading/trailing whitespace.
    """
    if not query:
        return ""
    # Use the more aggressive punctuation removal from the second snippet
    cleaned = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    # Add back explicit whitespace normalization for safety (from first snippet)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

@lru_cache(maxsize=128)
def is_greeting(query: str) -> bool:
    """
    Checks if the query is a simple greeting.
    A greeting is short and does not contain clear task-oriented keywords.
    """
    cleaned = clean_query(query)
    words = cleaned.split()

    # Check for direct matches in GREETINGS set
    if cleaned in GREETINGS:
        logging.info(f"✨ Detected exact greeting: '{query}'")
        return True

    # Check for longer greetings that don't contain task keywords (from first snippet logic)
    # Combined with the second snippet's focus on non-task words
    contains_task = any(keyword in cleaned for keyword in TASK_KEYWORDS)
    if not contains_task and len(words) <= 5 and any(g in cleaned for g in GREETINGS):
        logging.info(f"✨ Detected indirect greeting: '{query}'")
        return True

    return False

@lru_cache(maxsize=128)
def is_generic_query(query: str) -> bool:
    """
    Checks if the query is truly generic, non-task-oriented, or a meta-question.
    This function should return True only if the query cannot be classified as a greeting
    or a task-oriented request. This combines the robust logic from the second snippet
    with some pattern matching from the first.
    """
    if not query: # Added from first snippet for safety
        return True # A blank query could be considered generic

    q = clean_query(query)
    words = q.split()

    # Rule 1: Matches known generic phrases
    if q in GENERIC_INTENT_PHRASES:
        logging.info(f"❓ Detected generic (phrase match): '{query}'")
        return True

    # Rule 2: Pattern matching from the first snippet
    for pattern in GENERIC_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            logging.info(f"🔍 Generic (pattern match): {query}")
            return True

    # Rule 3: Very short queries (1-4 words) that are NOT greetings and contain NO task/formatting keywords.
    if 1 <= len(words) <= 4:
        if not is_greeting(query) and \
           not any(k in q for k in TASK_KEYWORDS) and \
           not any(k in q for k in FORMATTING_KEYWORDS):
            logging.info(f"❓ Detected generic (short, no task/format, not greeting): '{query}'")
            return True

    # Rule 4: Queries where all non-filler words are absent or very few, and no task/formatting keywords.
    non_filler_words = [w for w in words if w not in FILLER_WORDS]
    if len(non_filler_words) == 0: # All words are fillers
         if not is_greeting(query) and \
            not any(k in q for k in TASK_KEYWORDS) and \
            not any(k in q for k in FORMATTING_KEYWORDS):
            logging.info(f"❓ Detected generic (all fillers, no task/format, not greeting): '{query}'")
            return True

    # Rule 5: Partial phrase matching for longer queries (from first snippet logic)
    # Re-evaluating this; Rule 1/3/4 cover this better. Removing for simplicity as `q in GENERIC_INTENT_PHRASES` is more direct.
    # The new logic is more explicit about generic conditions.

    # If it contains any task or formatting keyword, it's NOT generic from the backend's perspective.
    if any(k in q for k in TASK_KEYWORDS) or any(k in q for k in FORMATTING_KEYWORDS):
        logging.info(f"🎯 Not generic (contains task or formatting keyword): '{query}'")
        return False

    return False

@lru_cache(maxsize=128)
def is_formatting_request(query: str) -> bool:
    """
    Checks if the query primarily asks for output formatting,
    considering if it's a standalone formatting request or part of a larger task.
    Uses refined logic from the second snippet.
    """
    if not query:
        return False
    cleaned = clean_query(query)
    words = cleaned.split()

    # Must contain at least one formatting keyword
    contains_format_keyword = any(k in cleaned for k in FORMATTING_KEYWORDS)
    if not contains_format_keyword:
        return False

    # Calculate ratio of formatting/filler words to total non-filler words
    relevant_words = [w for w in words if w not in FILLER_WORDS]
    formatting_relevant_words = [w for w in relevant_words if w in FORMATTING_KEYWORDS]

    # If a significant portion of non-filler words are formatting keywords, it's a primary formatting request
    if len(relevant_words) > 0 and len(formatting_relevant_words) / len(relevant_words) >= 0.75: # Increased threshold for primary formatting intent
        logging.info(f"⚙️ Formatting request (high formatting keyword density): '{query}'")
        return True

    # Simple check for very short, direct formatting requests
    if len(words) <= 3 and contains_format_keyword and not any(k in cleaned for k in TASK_KEYWORDS):
        logging.info(f"⚙️ Formatting request (very short & direct, no task): '{query}'")
        return True

    return False

@lru_cache(maxsize=128)
def contains_table_request(query: str) -> bool:
    """Detects if the query explicitly asks for a table format."""
    q = clean_query(query)
    # Combined terms from both snippets
    table_indicators = [
        "table", "tabular", "chart", "in a table", "in table format",
        "as a table", "spreadsheet", "grid format", "column", "row"
    ]
    return any(indicator in q for indicator in table_indicators)

@lru_cache(maxsize=128)
def contains_paragraph_request(query: str) -> bool:
    """Detects if the query explicitly asks for a paragraph format."""
    q = clean_query(query)
    return any(k in q for k in ["paragraph", "in paragraph form", "in paragraph"])

@lru_cache(maxsize=128)
def extract_diet_preference(query: str) -> str:
    """Extracts dietary preference (non-vegetarian, vegan, vegetarian, any)."""
    q = clean_query(query)
    if any(x in q for x in ["non-veg", "non veg", "nonvegetarian", "meat", "chicken", "fish", "egg"]):
        return "non-vegetarian"
    if "vegan" in q:
        return "vegan"
    if "veg" in q or "vegetarian" in q or "plant based" in q: # Added "plant based"
        return "vegetarian"
    return "any"

@lru_cache(maxsize=128)
def extract_diet_goal(query: str) -> str:
    """Extracts diet goal (weight loss, weight gain, general)."""
    q = clean_query(query)
    weight_loss_patterns = [
        "lose weight", "loss weight", "cut weight", "reduce weight",
        "lose fat", "cut fat", "slim down", "get lean", "shed pounds", "slimming", "lean" # Combined
    ]
    weight_gain_patterns = [
        "gain weight", "weight gain", "build muscle", "muscle gain",
        "mass gain", "bulk up", "get bigger", "add mass", "bulk", "bulking" # Combined
    ]

    if any(pattern in q for pattern in weight_loss_patterns):
        return "weight loss"
    if any(pattern in q for pattern in weight_gain_patterns):
        return "weight gain"

    # Specific checks from second snippet for "loss" or "gain" not in other contexts
    if "loss" in q and not any(kw in q for kw in ["data", "money", "time"]): # Avoid false positives
        return "weight loss"
    if "gain" in q and not any(kw in q for kw in ["knowledge", "experience"]): # Avoid false positives
        return "weight gain"

    return "general"

@lru_cache(maxsize=128)
def extract_regional_preference(query: str) -> str:
    """Extracts regional food preference from the query."""
    q = clean_query(query)
    for region in REGION_KEYWORDS:
        if region in q:
            # Use title for consistency, and handle "indian" explicitly
            if region == "indian":
                return "Indian"
            return region.title().replace("Indian", "").strip() or region.title() # Keep "Indian" for e.g. "South Indian"
    # If "indian" is explicitly mentioned but no specific region, default to "Indian"
    if "indian" in q:
        return "Indian"
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
    Combines phrases from both snippets.
    """
    q = clean_query(query)
    followup_phrases = [
        "same but", "make it", "change to", "instead", "now", "also",
        "can you", "could you", "would you", "will you", # From first snippet CONVERSATION_STARTERS
        "but make it", "what about", "and also", "next", "then", "continue",
        "how about", "modify", "update", "again", "for me", "just", # Added more from second snippet
        "different", "another", "alternative", "variation" # From first snippet
    ]

    # Check for direct follow-up phrases
    if any(p in q for p in followup_phrases):
        logging.info(f"🔄 Detected follow-up (phrase match): '{query}'")
        return True

    # Check for very short queries with some task keywords, implying modification
    words = q.split()
    non_filler_words = [w for w in words if w not in FILLER_WORDS]
    if len(words) <= 5 and len(non_filler_words) > 0 and len(non_filler_words) <= 3:
        if any(k in non_filler_words for k in TASK_KEYWORDS) or any(k in non_filler_words for k in FORMATTING_KEYWORDS):
             logging.info(f"🔄 Detected follow-up (short, context-modifying): '{query}'")
             return True

    return False

# --- Confidence Score (from first snippet, integrated into final metadata) ---
def get_query_confidence_score(query: str) -> float:
    """Calculate confidence score for query classification"""
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

    # Combine ratios for confidence score
    confidence = (task_ratio * 0.7) + (meaningful_ratio * 0.3)

    return min(confidence, 1.0)


def extract_all_metadata(query: str) -> Dict[str, Any]:
    """
    Extracts all relevant metadata from the query.
    The primary intent type helps determine the main category of the query
    for higher-level decision making in the backend.
    """
    logging.info(f"Starting analysis for query: '{query}'")

    # Clean query once to avoid repeated cleaning in sub-functions
    cleaned_q = clean_query(query)
    confidence = get_query_confidence_score(query) # Integrated confidence score

    # Determine primary intent type in a hierarchical manner (from second snippet)
    # 1. Greetings: Highest priority if it's purely a greeting
    if is_greeting(cleaned_q):
        primary_intent_type = "greeting"
    # 2. Formatting Requests: Next priority if it's primarily about output format
    #    and not primarily a task (e.g., "table please" vs "diet plan in table")
    elif is_formatting_request(cleaned_q) and not any(k in cleaned_q for k in TASK_KEYWORDS):
        primary_intent_type = "formatting"
    # 3. Generic Questions: If it doesn't fit greeting or formatting and isn't task-oriented
    elif is_generic_query(cleaned_q):
        primary_intent_type = "generic"
    # 4. Task-Oriented: Default if it contains relevant task keywords
    elif any(k in cleaned_q for k in TASK_KEYWORDS):
        primary_intent_type = "task"
    # 5. Fallback: If none of the above, it might still be treated as generic or minimal task
    else:
        primary_intent_type = "generic" # Or "unknown" if you want to be more explicit

    # Extract specific details only if it's relevant (i.e., not a pure greeting or generic meta-question)
    dietary_type = "any"
    goal = "general"
    region = "Indian"
    disease = None
    wants_table = False
    wants_paragraph = False
    is_follow_up = False
    has_task_keywords = any(keyword in cleaned_q for keyword in TASK_KEYWORDS) # From first snippet

    # If the primary intent is task, or if it's a formatting request that also contains task keywords,
    # then extract detailed diet-related metadata.
    if primary_intent_type == "task" or (primary_intent_type == "formatting" and has_task_keywords):
        dietary_type = extract_diet_preference(cleaned_q)
        goal = extract_diet_goal(cleaned_q)
        region = extract_regional_preference(cleaned_q)
        disease = extract_disease_condition(cleaned_q)
        is_follow_up = is_follow_up_query(cleaned_q) # Follow-up can apply to tasks and re-formatting of tasks

    # Formatting requests can apply to any query, but they are explicit
    wants_table = contains_table_request(cleaned_q)
    wants_paragraph = contains_paragraph_request(cleaned_q)

    # Special handling for queries like "give in paragraph form only"
    # If it's a pure formatting request, it might not contain other task keywords.
    # In such cases, the backend would typically apply this formatting to the *previous* context.
    if primary_intent_type == "formatting" and not is_follow_up and not has_task_keywords:
        # If it's purely a formatting request without task keywords, it's implicitly a follow-up on previous context
        is_follow_up = True
        # The other metadata (dietary_type, goal, region, disease) should ideally come from session memory
        # in the actual FastAPI app, not re-extracted from this pure formatting query.
        # Here, we just ensure the formatting flags are set correctly.

    metadata = {
        "primary_intent_type": primary_intent_type,
        "dietary_type": dietary_type,
        "goal": goal,
        "region": region,
        "disease": disease,
        "wants_table": wants_table,
        "wants_paragraph": wants_paragraph,
        "is_follow_up": is_follow_up,
        # Flags for raw classification results, can be useful for debugging
        "raw_is_greeting_flag": is_greeting(cleaned_q),
        "raw_is_generic_flag": is_generic_query(cleaned_q),
        "raw_is_formatting_request_flag": is_formatting_request(cleaned_q),
        "confidence_score": confidence, # From first snippet
        "query_length": len(query.split()) if query else 0, # From first snippet
        "has_task_keywords": has_task_keywords # From first snippet
    }
    logging.info(f"Final extracted metadata: {metadata}")
    return metadata

# --- Test Cases (for demonstration and understanding) ---
if __name__ == "__main__":
    test_queries = [
        "hi",                                         # Greeting
        "hello there",                                # Greeting
        "how are you",                                # Greeting (new)
        "what",                                       # Generic
        "who are you",                                # Generic
        "tell me about yourself",                     # Generic (new)
        "can you help",                               # Generic (new)
        "lol",                                        # Generic
        "?",                                          # Generic (pattern from first snippet)
        "ok",                                         # Generic (pattern from first snippet)
        "diet plan for weight loss",                  # Task
        "show me vegetarian diet",                    # Task with preferences
        "table format please",                        # Formatting only (should trigger follow_up = True implicitly)
        "make it vegan",                              # Follow-up + Preferences (short, context-modifying)
        "give me a diet plan",                        # Task
        "I want a vegan diet plan for weight loss",   # Task with preferences
        "suggest diet for diabetes in north indian style", # Task with disease and region
        "show a diet plan in table format",           # Task with formatting
        "give in table format",                       # Formatting only (should trigger follow_up = True implicitly)
        "in paragraph form only",                     # Formatting only (new, should trigger follow_up = True implicitly)
        "diet",                                       # Task (vague, but still a task)
        "healthy food",                               # Task (vague)
        "what about a diet for muscle gain",          # Follow-up + Task
        "same but make it non-veg",                   # Follow-up + Preferences
        "diet for jaisalmir",                         # Task with region
        "food for weight loss",                       # Task with goal
        "can you give me a plan for high bp",         # Task with disease (new)
        "I need a diet for thyroid",                  # Task with disease (new)
        "just table",                                 # Formatting only (should trigger follow_up = True implicitly)
        "plan for plant based diet",                  # Task with diet pref (new)
        "something for healthy heart",                # Task with disease (new)
        "give me some suggestions",                   # Task (vague, but asking for suggestions)
        "diet for me",                                # Task (vague)
        "in bullet points",                           # Formatting only (should trigger follow_up = True implicitly)
        "what can I eat"                              # Task (vague)
    ]

    print("--- Mixed and Consolidated Query Analysis Test Results ---")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        metadata = extract_all_metadata(q)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    print("\n--- End of Test Results ---")
