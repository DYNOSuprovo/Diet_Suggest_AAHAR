import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_vector_database(chroma_db_directory: str = "/tmp/chroma_db", in_memory: bool = False):
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

        # 🔍 DEBUG: Check how many docs exist
        try:
            count = len(db.get()['documents'])
            logging.info(f"📦 Vector DB loaded with {count} documents.")
        except Exception as e:
            logging.warning("⚠️ Could not count documents in Vector DB.")

        logging.info("✅ Chroma DB initialized successfully.")
        return db, embedding

    except Exception as e:
        logging.exception("❌ Vector DB setup failed.")
        raise
