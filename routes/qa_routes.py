from flask import Blueprint, request, jsonify
from services.qa_service import QAService

qa_bp = Blueprint("qa", __name__)
qa_service: QAService = None  # Will be injected later

@qa_bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = qa_service.answer(query)
    return jsonify({"results": results})

def init_qa_service(service: QAService):
    global qa_service
    qa_service = service
