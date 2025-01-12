"""
faiss_indexing.py

This module creates and manages a FAISS index for efficient data retrieval in the RAG pipeline.

Author: Satej
"""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_chunks(file_path: str) -> List[str]:
    """
    Loads preprocessed text chunks from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing preprocessed chunks.

    Returns:
        List[str]: List of text chunks.
    """
    df = pd.read_csv(file_path)
    return df['chunk'].tolist()


def create_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Creates embeddings for the input text chunks using a sentence transformer model.

    Parameters:
        chunks (List[str]): List of text chunks.
        model_name (str): Hugging Face sentence transformer model name.

    Returns:
        np.ndarray: Array of embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)


def build_faiss_index(embeddings: np.ndarray, output_path: str) -> None:
    """
    Builds a FAISS index from embeddings and saves it.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.
        output_path (str): Path to save the FAISS index file.

    Returns:
        None
    """
    # Create a FAISS index with cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, output_path)
    print(f"FAISS index saved to {output_path}")


def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Loads a saved FAISS index from file.

    Parameters:
        index_path (str): Path to the saved FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index.
    """
    index = faiss.read_index(index_path)
    return index


if __name__ == "__main__":
    chunk_file = "/home/satej/data/processed_chunks.csv"  # Example chunk file path
    index_file = "/home/satej/models/faiss_index.bin"  # Example FAISS index path

    chunks = load_chunks(chunk_file)
    embeddings = create_embeddings(chunks)
    build_faiss_index(embeddings, index_file)

    # Example of loading the index for use
    index = load_faiss_index(index_file)
    print("FAISS index loaded successfully.")
