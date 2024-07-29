#!/usr/bin/env python3

import logging
import os
import json
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, load_metric
import torch

# Disable wandb integration
class NoWandbCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        args.report_to = []

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
            report_to=[]  # Disable reporting to wandb
        )
        logger.info(f"ModelTrainer initialized using device: {self.device}")

    def tokenize_function(self, examples):
        inputs = self.tokenizer(examples['query'], padding="max_length", truncation=True, max_length=512)
        targets = self.tokenizer(examples['response'], padding="max_length", truncation=True, max_length=512)
        inputs['labels'] = targets['input_ids']
        return inputs

    def train_on_data(self, data):
        dataset = Dataset.from_dict(data)
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        # Initialize the Trainer with the NoWandbCallback
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_datasets,
            callbacks=[NoWandbCallback]
        )

        # Train the model
        trainer.train()
        logger.info("Model training completed.")

        # Save the model and tokenizer
        self.model.save_pretrained("./model")
        self.tokenizer.save_pretrained("./model")

    def train_models_from_json(self, data_dir):
        logger.info(f"Training models using JSON files in {data_dir}...")
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    self.train_on_data(data)

    def update_json_file(self, file_path, data):
        for example in data['examples']:
            response = example['response']
            response_embedding = self.sentence_transformer.encode(response)
            example['embedding'] = response_embedding.tolist()

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models using JSON data files")
    parser.add_argument('--data-dir', type=str, default='training_data', help='Directory containing JSON data files for training')
    args = parser.parse_args()

    trainer = ModelTrainer()
    trainer.train_models_from_json(args.data_dir)
