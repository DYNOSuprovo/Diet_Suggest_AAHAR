import os
import logging
import gdown
import zipfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_DRIVE_FILE_ID = "1uVdkPNgcl_QJ5j5i3kM-7egGTy65E4oB"  # replace with your file ID
DEST_ZIP = "vector_db.zip"
EXTRACT_DIR = "db"

def download_prebuilt_chroma_db_if_missing():
    """Downloads and extracts prebuilt Chroma DB if not already present."""
    if os.path.exists(EXTRACT_DIR) and os.path.isdir(EXTRACT_DIR):
        logging.info("✅ Chroma vector DB already exists locally. Skipping download.")
        return

    try:
        logging.info("📥 Prebuilt vector DB not found. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, DEST_ZIP, quiet=False)

        with zipfile.ZipFile(DEST_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        
        os.remove(DEST_ZIP)
        logging.info("✅ Vector DB downloaded and extracted successfully.")
    except Exception as e:
        logging.exception("❌ Failed to download or extract the vector DB.")
        raise RuntimeError("Error downloading or extracting Chroma DB") from e

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
        logging.info("✅ Gemini Embeddings loaded.")

        if not in_memory:
            download_prebuilt_chroma_db_if_missing()

        db = Chroma(
            persist_directory=None if in_memory else chroma_db_directory,
            embedding_function=embedding
        )
        logging.info("✅ Chroma DB initialized successfully.")
        return db, embedding

    except FileNotFoundError as fnf_error:
        logging.error(str(fnf_error))
        raise
    except Exception as e:
        logging.exception("Vector DB setup failed due to unexpected error.")
        raise
