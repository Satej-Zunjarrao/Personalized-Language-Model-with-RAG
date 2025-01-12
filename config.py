"""
config.py

This module centralizes configuration variables and constants for the RAG pipeline project.

Author: Satej
"""

# Paths for data and models
RAW_DATA_PATH = "/home/satej/data/raw_data.csv"
PROCESSED_DATA_PATH = "/home/satej/data/processed_chunks.csv"
QA_PAIRS_PATH = "/home/satej/data/qa_pairs.csv"
FAISS_INDEX_PATH = "/home/satej/models/faiss_index.bin"
FINE_TUNED_MODEL_PATH = "/home/satej/models/fine_tuned_model"
LOG_FILE_PATH = "/home/satej/logs/rag_pipeline.log"

# Model configurations
DEFAULT_MODEL_NAME = "t5-small"  # Pretrained Hugging Face model name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer for embeddings

# Preprocessing configurations
CHUNK_SIZE = 5  # Number of sentences per chunk
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for tokenization

# Training configurations
TRAIN_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# API configurations
API_HOST = "0.0.0.0"
API_PORT = 5000

# Logging configurations
LOGGING_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
