from flask import Flask
from routes.qa_routes import qa_bp, init_qa_service
from services.pdf_service import PDFService
from services.vector_store_service import VectorStoreService
from services.qa_service import QAService
from config.settings import settings
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.register_blueprint(qa_bp)

vs_service = VectorStoreService()

pdf_service = PDFService(settings.PDF_FILE)
structured_data = pdf_service.extract_with_structure()

if not vs_service.exists():
    print("Vector store not found. Creating...")
    store = vs_service.build_store(
        paragraphs_with_meta=structured_data["paragraphs"],
        headings=structured_data["headings"],
        subheadings=structured_data["subheadings"]
    )
else:
    print("Loading existing vector store...")
    store = vs_service.load_store()

# Pass both FAISS store and structured_data to QAService
qa_service_instance = QAService(store, structured_data["paragraphs"])
init_qa_service(qa_service_instance)

if __name__ == "__main__":
    app.run(debug=True)
