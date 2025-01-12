"""
rag_pipeline.py

This module implements the RAG pipeline by combining a retrieval system with a fine-tuned 
language model to generate personalized responses.

Author: Satej
"""

import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    """
    Implements a Retrieval-Augmented Generation (RAG) pipeline.
    """
    def __init__(self, index_path: str, model_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initializes the RAG pipeline with the retrieval index and language model.

        Parameters:
            index_path (str): Path to the FAISS index file.
            model_path (str): Path to the fine-tuned language model directory.
            embedding_model (str): Name of the sentence transformer for embeddings.
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print("FAISS index loaded successfully.")

        # Load sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(embedding_model)

        # Load fine-tuned language model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def retrieve_context(self, query: str, top_k: int = 5) -> list:
        """
        Retrieves top-k relevant contexts for a given query using FAISS.

        Parameters:
            query (str): The user query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: List of top-k retrieved contexts.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Perform FAISS search
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0].tolist()

    def generate_response(self, query: str, contexts: list) -> str:
        """
        Generates a response using the language model with retrieved contexts.

        Parameters:
            query (str): The user query.
            contexts (list): Retrieved contexts as input.

        Returns:
            str: Generated response.
        """
        # Concatenate query and retrieved contexts
        input_text = f"Query: {query} Context: {' '.join(contexts)}"

        # Tokenize and generate response
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example paths
    index_file = "/home/satej/models/faiss_index.bin"
    model_dir = "/home/satej/models/fine_tuned_model"
    query_example = "What is the warranty policy for this product?"

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(index_file, model_dir)

    # Retrieve context and generate response
    retrieved_indices = rag_pipeline.retrieve_context(query_example)
    retrieved_contexts = ["Example context 1", "Example context 2"]  # Replace with actual contexts
    response = rag_pipeline.generate_response(query_example, retrieved_contexts)

    print("Generated Response:", response)
