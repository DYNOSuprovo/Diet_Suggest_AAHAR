# config.py
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_environment_variables():
    """Loads environment variables and returns API keys."""
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    return gemini_api_key, groq_api_key

def configure_api_keys(gemini_api_key: str, groq_api_key: str):
    """Configures Google Generative AI and checks API keys."""
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
        logging.error("GEMINI_API_KEY not found.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    try:
        genai.configure(api_key=gemini_api_key)
        logging.info("Google Generative AI configured successfully.")
    except Exception as e:
        st.error(f"Failed to configure Google Generative AI: {e}")
        logging.error(f"Google GenAI Configuration Error: {e}")
        st.stop()

    if not groq_api_key:
        st.warning("GROQ_API_KEY not found. Groq suggestions will be unavailable.")
        logging.warning("GROQ_API_KEY not found.")

    return gemini_api_key, groq_api_key