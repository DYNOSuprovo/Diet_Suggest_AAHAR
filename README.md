# 🥗 AI Diet Suggestion System (Indian Edition)

> Personalized Indian dietary advice using LangChain, Gemini, Groq, and ChromaDB

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-success?logo=OpenAI)
![Gemini API](https://img.shields.io/badge/Gemini-Pro_API-orange?logo=google)
![Groq](https://img.shields.io/badge/Groq-LLaMA3%2FMixtral%2FGemma-blueviolet?logo=groq)
![ChromaDB](https://img.shields.io/badge/Vectorstore-ChromaDB-green?logo=database)
![Status](https://img.shields.io/badge/Backend-FastAPI_Ready-brightgreen)
![License](https://img.shields.io/github/license/DYNOSuprovo/Diet_Suggest_AAHAR)

---

## 📸 Project Overview

**AI Diet Suggestion System** intelligently provides regionally-aware and dietary-type-specific Indian food suggestions using a RAG (Retrieval-Augmented Generation) pipeline, conversational memory, and fallback LLM integrations via Groq.

It understands queries like:

> *"Suggest a South Indian vegetarian dinner plan for diabetes."*
> And returns contextually aware meal advice grounded in nutritional logic and culturally suitable food habits.

---

## 🧠 Architecture Workflow

![Architecture Flowchart](A_flowchart_diagram_in_the_image_illustrates_a_die.png)

---

### 🯩 Workflow Explanation

1. **User Query** — Input via chatbot or frontend (e.g., "high-protein vegetarian diet for muscle gain").
2. **Session Memory** — Previous chats are fetched using `ChatMessageHistory` for continuity.
3. **Context Retrieval** — The system queries a ChromaDB vectorstore for relevant dietary documents.
4. **Prompt Construction** — Gemini receives a dynamic prompt with context, chat history, and user metadata (diet type, region, goal).
5. **LLM Response** — Gemini generates a food plan or suggestion. If it's too generic or an error, backup suggestions are fetched from:

   * Groq’s **LLaMA3**, **Mixtral**, and **Gemma** models in parallel.
6. **Merge Logic** — Responses are merged using smart prompt templates for a final, coherent answer (text or markdown table).
7. **Output** — Final result is streamed/displayed to user.

---

## 📂 File Overview

| File                  | Role                                                                         |
| --------------------- | ---------------------------------------------------------------------------- |
| `llm_chains.py`       | Core logic for LangChain pipeline: RAG + Gemini + Prompt Templates + History |
| `groq_integration.py` | Multi-threaded Groq API call handler using `requests`                        |
| `chroma_db_loader.py` | (Optional) Script to ingest and persist documents into Chroma vectorstore    |
| `fastapi_app.py`      | REST API backend to expose endpoints for frontend (if applicable)            |
| `requirements.txt`    | Python dependencies                                                          |

---

## 🛠 Tech Stack

| Component  | Tool                                              |
| ---------- | ------------------------------------------------- |
| Language   | Python 3.11                                       |
| Framework  | LangChain                                         |
| LLM APIs   | Google Gemini Pro, Groq (LLaMA3, Mixtral, Gemma2) |
| Vector DB  | ChromaDB                                          |
| Backend    | FastAPI (optional)                                |
| Deployment | Local / Render / Hugging Face Spaces              |

---

## 🚀 User Guide

### 🔧 Setup

```bash
# Clone repository
git clone https://github.com/DYNOSuprovo/Diet_Suggest_AAHAR.git
cd Diet_Suggest_AAHAR

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 🔑 Required API Keys

Create a `.env` file or set environment variables:

```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

---

### 🥪 Run Backend (if using FastAPI)

```bash
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

Then visit: `http://localhost:8000/docs` for Swagger API.

---

### 🧪 Local Example (No FastAPI)

```python
from llm_chains import setup_qa_chain, setup_conversational_qa_chain
from groq_integration import cached_groq_answers

# Setup Gemini + Chroma + prompt
qa_chain = setup_qa_chain(llm_gemini, db, define_rag_prompt_template())
conversation = setup_conversational_qa_chain(qa_chain)

response = conversation.invoke({
    "query": "Suggest a fiber-rich Indian dinner for constipation",
    "dietary_type": "vegetarian",
    "goal": "digestive health",
    "region": "North India"
}, config={"configurable": {"session_id": "user-123"}})

print(response)
```

---

## 📊 Output Types

* ✅ Plain-text RAG-based answer
* ✅ Merged answer using Groq if needed
* ✅ Markdown table (if selected)
* ✅ Culturally tuned to Indian meals (e.g., poha, khichdi, ragi)

---

## 🧠 LLM Logic

* **Primary LLM**: Gemini (context-aware)
* **Backup**: Groq (LLaMA3, Mixtral, Gemma2)
* **Merge Templates**:

  * Default: for clean answers
  * Table Mode: for markdown diet plan tables

---

## 🚧 Future Improvements

* 📱 Flutter-based frontend (UI for querying)
* 📄 PDF export of diet plans
* 📬 WhatsApp/Telegram bot integration
* 🗂️ Caching with Redis for response optimization
* 📟 Food item-level calorie estimates

---

## 🙏 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [Gemini API](https://ai.google.dev/)
* [Groq API](https://console.groq.com/)
* [Chroma](https://www.trychroma.com/)

---

## 📜 License

MIT License — Fork it, use it, contribute!
