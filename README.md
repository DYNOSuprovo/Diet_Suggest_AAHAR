# 🥗 AAHAR: Advanced Assistant for Healthy Alimentary Recommendations

AAHAR is a sophisticated AI-driven nutrition assistant tailored for Indian dietary needs. It leverages Retrieval-Augmented Generation (RAG), a comprehensive Indian food nutrition database, and multi-model LLM orchestration to provide personalized diet plans, meal analysis, and weather-aware food suggestions.

## 🚀 Key Features

*   **Intelligent Chat Assistant**: Personalized Indian diet advice using RAG and a curated food database.
*   **Meal Analysis**: Input your meal items (e.g., "Oatmeal with almonds"), and AAHAR calculates total calories, protein, carbs, and provides a nutritional audit.
*   **Nutrition Database Search**: Instant lookup for over 3,500+ Indian food items with macro-nutrient details.
*   **Weather-Aware Recommendations**: Suggests cooling foods in summer and warming foods in winter based on your local weather.
*   **Multi-Model Orchestration**: Combines the reasoning capabilities of Groq (LLaMA 3) and Gemini for high-quality, balanced responses.

## 🛠️ Technology Stack

*   **Backend**: FastAPI, LangChain, Pydantic, Uvicorn.
*   **AI/LLM**: Groq (LLaMA 3.3 70B), Gemini Pro (Fallback), Google Generative AI Embeddings.
*   **Vector Store**: ChromaDB (Retrieval of dietary guidelines).
*   **Data Science**: Pandas, FuzzyWuzzy (Nutrition database search).

## 📁 Project Structure

```text
Diet_Suggest_AAHAR-main/
├── app/
│   ├── api/                # API Route Handlers
│   │   ├── chat.py         # /api/chat endpoint
│   │   ├── meal_analysis.py # /api/analyze-meal endpoint
│   │   ├── nutrition.py    # /api/nutrition search/stats
│   │   └── utilities.py    # /api/health and other utilities
│   ├── core/
│   │   └── globals.py      # App-wide state (LLMs, Vector DB)
│   ├── agent_tools.py      # Tools for the AI Agent (Weather, Recipe, etc)
│   ├── llm_chains.py       # LangChain chain definitions and RAG logic
│   ├── models.py           # Pydantic models (Request/Response schemas)
│   ├── nutrition_search.py # Search logic for 3,500+ food records
│   ├── prompts.py          # AI Prompt Templates
│   ├── query_analysis.py   # Intent extraction and sentiment analysis
│   └── vector_store.py     # ChromaDB management and HuggingFace sync
├── fastapi_app6.py         # Main entry point and server initialization
├── nutrition_data.json     # Curated Indian food nutrition dataset
└── .env                    # Environment variables (API Keys)
```

## ⚙️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/aahar.git
    cd aahar
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file with the following keys:
    ```env
    GEMINI_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key
    OPENWEATHER_API_KEY=your_weather_key
    FASTAPI_SECRET_KEY=your_long_secret_string
    ```

4.  **Run the Server**:
    ```bash
    python fastapi_app6.py
    ```
    The server will automatically download the required Vector DB on first startup.

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Main chat endpoint (Agentic RAG) |
| `POST` | `/api/analyze-meal` | Nutrition analysis of a list of dishes |
| `GET` | `/api/nutrition/search/{food}` | Direct search in the food database |
| `GET` | `/api/health` | Check system and component status |

## 👨‍💻 Author
Created by **Suprovo**.
Dedicated to making balanced nutrition accessible through AI.
