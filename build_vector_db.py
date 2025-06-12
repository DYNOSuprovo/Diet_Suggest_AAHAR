import os
import gdown
import zipfile
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

def download_prebuilt_db():
    db_folder = "db"
    zip_path = "db.zip"

    if not os.path.exists(db_folder):
        print("📥 DB not found. Downloading prebuilt vector store...")
        # ✅ Your actual Google Drive File ID
        file_id = "1FiUvNdx9mVNpk1Mek5SAzezPGpJIwu5-"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, zip_path, quiet=False)

        print("📦 Extracting DB zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(db_folder)

        print("✅ Prebuilt DB downloaded and extracted.")
    else:
        print("✅ Existing DB already present. Skipping download.")

def build_vector_db_from_pdfs():
    print("🔧 Building vector store from PDFs...")
    loader = DirectoryLoader("pdfs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents, embedding=embedding, persist_directory="db")
    db.persist()
    print("✅ Vector store created from PDFs.")

if __name__ == "__main__":
    download_prebuilt_db()
    # Optional: enable this line to rebuild DB from PDFs even if DB exists
    # build_vector_db_from_pdfs()
