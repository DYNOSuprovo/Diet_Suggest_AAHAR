# embedding_utils.py
import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_vector_database(chroma_db_directory: str = "db", in_memory: bool = False):
    """
    Initializes Chroma vector database using Gemini embeddings.

    Parameters:
        chroma_db_directory (str): Path to the Chroma DB directory.
        in_memory (bool): If True, does not persist the DB to disk.

    Returns:
        db: Chroma vector store instance
        embedding: Gemini embedding function
    """
    try:
        logging.info("Initializing Gemini Embeddings for vector DB...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment variables.")
        
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        logging.info("Gemini Embeddings loaded successfully.")

        if not in_memory and not os.path.exists(chroma_db_directory):
            raise FileNotFoundError(f"ChromaDB directory '{chroma_db_directory}' not found.")

        db = Chroma(
            persist_directory=None if in_memory else chroma_db_directory,
            embedding_function=embedding
        )
        logging.info("Chroma DB initialized successfully.")
        return db, embedding

    except FileNotFoundError as fnf_error:
        logging.error(str(fnf_error))
        raise
    except Exception as e:
        logging.exception("Vector DB setup failed due to unexpected error.")
        raise
