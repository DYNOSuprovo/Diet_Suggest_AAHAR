from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os, logging

def setup_vector_database(chroma_db_directory: str = "db"):
    try:
        logging.info("Initializing lightweight Gemini Embeddings for Render.")

        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
        logging.info("Google GenerativeAI Embeddings initialized successfully.")

        if not os.path.exists(chroma_db_directory):
            logging.error(f"ChromaDB directory '{chroma_db_directory}' not found.")
            raise FileNotFoundError(f"ChromaDB directory '{chroma_db_directory}' not found.")

        db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding)
        logging.info("Chroma DB loaded successfully.")
        return db, embedding

    except Exception as e:
        logging.exception("VectorDB setup failed:")
        raise
