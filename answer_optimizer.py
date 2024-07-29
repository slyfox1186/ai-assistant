#!/usr/bin/env python3

import sqlite3
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, Trainer, TrainingArguments
from torch.utils.data import Dataset
from regex_cleaner import RegexCleaner
from database_handler import DatabaseHandler

# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AnswerDataset(Dataset):
    def __init__(self, queries, responses, tokenizer, max_length=512):
        self.queries = queries
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]
        inputs = self.tokenizer.encode_plus(query, response, add_special_tokens=True, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask

class AnswerOptimizer:
    def __init__(self, db_path='answer_optimizer.db'):
        self.model_name = 'facebook/bart-large-cnn'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
        self.db_path = db_path
        self.db_handler = DatabaseHandler()
        self.regex_cleaner = RegexCleaner(exclusion_files=self.db_handler.exclusion_files)
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS Answers
                              (id INTEGER PRIMARY KEY, query TEXT, original TEXT, optimized TEXT, sentiment TEXT, timestamp REAL)''')
            cursor.execute("PRAGMA table_info(Answers)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'sentiment' not in columns:
                cursor.execute('ALTER TABLE Answers ADD COLUMN sentiment TEXT')
            if 'query' not in columns:
                cursor.execute('ALTER TABLE Answers ADD COLUMN query TEXT')
            conn.commit()

    def _store_data(self, query: str, original: str, optimized: str, sentiment: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO Answers (query, original, optimized, sentiment, timestamp) VALUES (?, ?, ?, ?, ?)',
                           (query, original, optimized, sentiment, time.time()))
            conn.commit()

    def optimize_answer(self, query: str, answer: str) -> str:
        cleaned_answer = self.regex_cleaner.clean_text(answer)
        if len(cleaned_answer) == 0:
            logger.error("Cleaned answer is empty after regex processing.")
            return answer
        input_length = len(self.tokenizer.encode(cleaned_answer, add_special_tokens=False))
        max_length = min(150, input_length)
        min_length = min(30, max_length - 1)
        optimized = self.summarizer(cleaned_answer, max_length=max_length, min_length=min_length, do_sample=False)
        optimized_answer = optimized[0]['summary_text']
        sentiment = self.analyze_sentiment(optimized_answer)
        self._store_data(query, cleaned_answer, optimized_answer, sentiment)
        return optimized_answer

    def analyze_sentiment(self, text: str) -> str:
        analysis = self.sentiment_analyzer(text)
        return analysis[0]['label']

    def train(self, query: str, optimized_answer: str):
        logger.info(f"Training the model with query: {query}")
        logger.info(f"Optimized answer: {optimized_answer}")

        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        labels = self.tokenizer(optimized_answer, return_tensors="pt", truncation=True, max_length=512)

        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100

        train_dataset = torch.utils.data.TensorDataset(inputs["input_ids"].to(self.device), labels["input_ids"].to(self.device))

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        print("Model training completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer Optimization Script")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to get an answer for")
    parser.add_argument("--answer", "-a", type=str, required=True, help="The answer to optimize")
    args = parser.parse_args()

    optimizer = AnswerOptimizer()
    optimized_answer = optimizer.optimize_answer(args.query, args.answer)
    print(optimized_answer)