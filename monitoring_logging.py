"""
monitoring_logging.py

This module provides utilities for monitoring and logging query handling performance
in the RAG pipeline, including response accuracy and retrieval latency.

Author: Satej
"""

import logging
import time
from typing import Callable


# Configure logging
LOG_FILE = "/home/satej/logs/rag_pipeline.log"  # Example log file path
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_query(query: str, response: str, contexts: list, latency: float):
    """
    Logs query information, response, retrieved contexts, and response time.

    Parameters:
        query (str): The user's query.
        response (str): The generated response.
        contexts (list): Retrieved contexts.
        latency (float): Time taken to process the query in seconds.

    Returns:
        None
    """
    logging.info(
        f"Query: {query}\n"
        f"Response: {response}\n"
        f"Retrieved Contexts: {contexts}\n"
        f"Latency: {latency:.2f} seconds\n"
        "-------------------------------"
    )


def monitor_performance(func: Callable):
    """
    Decorator to monitor and log the execution time of a function.

    Parameters:
        func (Callable): The function to monitor.

    Returns:
        Callable: Wrapped function with monitoring.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time

        # Log function execution time
        logging.info(
            f"Function: {func.__name__}\n"
            f"Execution Time: {latency:.2f} seconds\n"
            "-------------------------------"
        )
        return result

    return wrapper


# Example usage in RAG pipeline
if __name__ == "__main__":
    from rag_pipeline import RAGPipeline

    # Example query
    query_example = "What is the warranty policy for this product?"

    # Initialize RAG pipeline
    index_file = "/home/satej/models/faiss_index.bin"
    model_dir = "/home/satej/models/fine_tuned_model"
    rag_pipeline = RAGPipeline(index_file, model_dir)

    @monitor_performance
    def handle_query(query):
        # Retrieve contexts
        retrieved_indices = rag_pipeline.retrieve_context(query)
        retrieved_contexts = [
            "Example context 1", "Example context 2"
        ]  # Replace with actual retrieval logic

        # Generate response
        response = rag_pipeline.generate_response(query, retrieved_contexts)

        # Log the query
        latency = time.time() - start_time
        log_query(query, response, retrieved_contexts, latency)
        return response

    # Simulate query handling
    response = handle_query(query_example)
    print("Generated Response:", response)
