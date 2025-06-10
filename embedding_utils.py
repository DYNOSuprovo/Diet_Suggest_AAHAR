# embedding_utils.py

import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_vector_database(chroma_db_directory: str = "db"):
    """
    Sets up and returns the Chroma vector store and embedding function.
    Logs and raises errors if setup fails.
    """
    try:
        logging.info("Initializing lightweight HuggingFaceEmbeddings for Render.")
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': True}
        )

        if not os.path.exists(chroma_db_directory):
            raise FileNotFoundError(
                f"ChromaDB directory '{chroma_db_directory}' not found. Please ensure the DB is initialized."
            )

        db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding)
        logging.info("Chroma DB loaded successfully.")
        return db, embedding

    except Exception as e:
        logging.exception(f"VectorDB setup error: {e}")
        raise RuntimeError(f"Vector DB initialization failed: {e}")
