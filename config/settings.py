import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PDF_FILE = os.getenv("PDF_FILE", "documents/document.pdf")
    VECTOR_FOLDER = os.getenv("VECTOR_FOLDER", "vector_store")

settings = Settings()
