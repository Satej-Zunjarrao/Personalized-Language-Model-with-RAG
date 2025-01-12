"""
data_preprocessing.py

This module handles data preprocessing for the RAG pipeline. It includes functions
to clean, tokenize, and segment domain-specific text into chunks suitable for retrieval.

Author: Satej
"""

import pandas as pd
import re
from typing import List


def clean_text(text: str) -> str:
    """
    Cleans input text by removing unwanted characters, extra spaces, and special symbols.

    Parameters:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Remove special characters and multiple spaces
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes the input text into sentences using basic splitting.

    Parameters:
        text (str): The cleaned text.

    Returns:
        List[str]: List of sentences from the input text.
    """
    # Split text by periods to approximate sentence tokenization
    return [sentence.strip() for sentence in text.split(".") if sentence.strip()]


def segment_text(sentences: List[str], chunk_size: int = 5) -> List[str]:
    """
    Segments tokenized sentences into chunks for efficient retrieval.

    Parameters:
        sentences (List[str]): List of tokenized sentences.
        chunk_size (int): Number of sentences per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = [
        " ".join(sentences[i:i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]
    return chunks


def preprocess_data(input_file: str, output_file: str) -> None:
    """
    Main function to preprocess the input data file and save the processed chunks.

    Parameters:
        input_file (str): Path to the raw input text file.
        output_file (str): Path to save the processed chunks as a CSV file.

    Returns:
        None
    """
    # Load the data
    df = pd.read_csv(input_file)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['tokenized_sentences'] = df['cleaned_text'].apply(tokenize_text)
    df['chunks'] = df['tokenized_sentences'].apply(segment_text)

    # Flatten the chunks and save them to a new DataFrame
    processed_chunks = pd.DataFrame(
        [chunk for chunks in df['chunks'] for chunk in chunks],
        columns=['chunk']
    )

    # Save the processed chunks to a CSV file
    processed_chunks.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    input_file = "/home/satej/data/raw_data.csv"  # Example input path
    output_file = "/home/satej/data/processed_chunks.csv"  # Example output path
    preprocess_data(input_file, output_file)
