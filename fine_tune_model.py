"""
fine_tune_model.py

This module fine-tunes a pre-trained language model on domain-specific QA pairs to enhance 
contextual understanding and response generation.

Author: Satej
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset


def load_data(file_path: str) -> Dataset:
    """
    Loads domain-specific QA data for fine-tuning from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing QA pairs.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    data = load_dataset("csv", data_files=file_path)
    return data["train"]


def preprocess_data(data: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenizes and processes the QA data for model input.

    Parameters:
        data (Dataset): Raw dataset with questions and answers.
        tokenizer: Tokenizer for the language model.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        Dataset: Tokenized dataset.
    """
    def tokenize_function(example):
        return tokenizer(
            example["question"], example["answer"],
            truncation=True, max_length=max_length, padding="max_length"
        )

    return data.map(tokenize_function, batched=True)


def fine_tune_model(
    model_name: str,
    train_dataset: Dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """
    Fine-tunes a pre-trained model on the processed QA data.

    Parameters:
        model_name (str): Hugging Face model name (e.g., "t5-small").
        train_dataset (Dataset): Tokenized dataset for training.
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        None
    """
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        evaluation_strategy="no",
        weight_decay=0.01
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")


if __name__ == "__main__":
    data_file = "/home/satej/data/qa_pairs.csv"  # Example QA pairs path
    model_output = "/home/satej/models/fine_tuned_model"  # Example output directory
    model_name = "t5-small"  # Replace with desired model name (e.g., GPT-2)

    # Load and preprocess data
    raw_data = load_data(data_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processed_data = preprocess_data(raw_data, tokenizer)

    # Fine-tune the model
    fine_tune_model(model_name, processed_data, model_output)
