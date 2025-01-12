"""
utils.py

This module provides utility functions for shared operations across the RAG pipeline project.

Author: Satej
"""

import os
import logging


def ensure_directory_exists(directory: str):
    """
    Ensures that the given directory exists, creating it if necessary.

    Parameters:
        directory (str): Path to the directory.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")


def calculate_cosine_similarity(vector1, vector2):
    """
    Calculates cosine similarity between two vectors.

    Parameters:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    norm1 = sum(v1 ** 2 for v1 in vector1) ** 0.5
    norm2 = sum(v2 ** 2 for v2 in vector2) ** 0.5
    return dot_product / (norm1 * norm2)


def log_message(message: str, level: str = "INFO"):
    """
    Logs a message to the console and log file at the specified level.

    Parameters:
        message (str): The message to log.
        level (str): Logging level (e.g., INFO, DEBUG, WARNING).

    Returns:
        None
    """
    levels = {
        "DEBUG": logging.debug,
        "INFO": logging.info,
        "WARNING": logging.warning,
        "ERROR": logging.error,
    }
    log_function = levels.get(level.upper(), logging.info)
    log_function(message)


if __name__ == "__main__":
    # Example usage
    ensure_directory_exists("/home/satej/logs")
    print("Cosine Similarity:", calculate_cosine_similarity([1, 2, 3], [4, 5, 6]))
    log_message("This is a test log message.", "INFO")
