# embedding_utils.py
import os
import streamlit as st
import logging
from langchain_community.vectorstores import Chroma  # <--- THIS LINE IS CRUCIAL
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_vector_database(chroma_db_directory: str = "db"):
    """
    Sets up and returns the ChromaDB vector store and embedding function.
    Stops the Streamlit app if setup fails.
    """
    try:
        logging.info("Attempting to load SentenceTransformer model for embeddings.")
        try:
            # Check if the model is available/downloadable
            SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("SentenceTransformer model 'all-MiniLM-L6-v2' is available.")
        except Exception as model_e:
            st.error(f"Failed to load SentenceTransformer model: {model_e}. Please check your internet connection or environment.")
            logging.error(f"SentenceTransformer model loading error: {model_e}")
            st.stop()

        embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': True}
        )
        logging.info("HuggingFaceEmbeddings initialized successfully.")

        if not os.path.exists(chroma_db_directory):
            st.error(f"ChromaDB directory '{chroma_db_directory}' not found. Please ensure the DB is initialized first.")
            logging.error(f"ChromaDB directory '{chroma_db_directory}' not found.")
            st.stop()

        db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding)
        logging.info("Chroma DB loaded successfully.")
        return db, embedding

    except Exception as e:
        st.error(f"VectorDB setup error: {e}")
        logging.exception("Full VectorDB setup traceback:")
        st.stop()