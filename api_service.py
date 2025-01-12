"""
api_service.py

This module deploys the RAG pipeline as a REST API using Flask, allowing real-time
query handling and personalized response generation.

Author: Satej
"""

from flask import Flask, request, jsonify
from rag_pipeline import RAGPipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize RAG pipeline
INDEX_FILE = "/home/satej/models/faiss_index.bin"  # Example FAISS index path
MODEL_DIR = "/home/satej/models/fine_tuned_model"  # Example fine-tuned model path
rag_pipeline = RAGPipeline(INDEX_FILE, MODEL_DIR)


@app.route("/api/query", methods=["POST"])
def handle_query():
    """
    Handles incoming POST requests for query processing.

    Request JSON Format:
        {
            "query": "User's query string"
        }

    Response JSON Format:
        {
            "response": "Generated response from the RAG pipeline",
            "retrieved_contexts": ["Context 1", "Context 2", ...]
        }

    Returns:
        JSON response with generated answer and retrieved contexts.
    """
    # Extract query from request
    request_data = request.json
    query = request_data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        # Retrieve contexts
        retrieved_indices = rag_pipeline.retrieve_context(query)
        retrieved_contexts = [
            "Example context 1", "Example context 2"
        ]  # Replace with actual retrieval logic if needed

        # Generate response
        response = rag_pipeline.generate_response(query, retrieved_contexts)

        # Return response as JSON
        return jsonify({"response": response, "retrieved_contexts": retrieved_contexts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
