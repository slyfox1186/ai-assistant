#!/usr/bin/env python3

import logging
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from database_handler import DatabaseHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to train the model
def train_model(model_name: str = "facebook/bart-large-cnn", dataset_name: str = "cnn_dailymail", config: str = "3.0.0", db_path: str = "ai_assistant.db"):
    logger.info(f"Starting model training for {model_name}...")
    
    # Load the dataset with a specified config
    dataset = load_dataset(dataset_name, config)

    # Load model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Define maximum length for padding and truncation
    max_length = 1024

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples['article'], padding="max_length", truncation=True, max_length=max_length)
        targets = tokenizer(examples['highlights'], padding="max_length", truncation=True, max_length=max_length)
        inputs['labels'] = targets['input_ids']
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        report_to=[]
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )

    # Train the model
    trainer.train()
    logger.info("Model training completed.")

    # Save the trained model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    
    # Save the trained model to the database
    db_handler = DatabaseHandler(db_path)
    db_handler.save_model(model)
    logger.info("Trained model saved to the database.")

if __name__ == "__main__":
    train_model()