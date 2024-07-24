#!/usr/bin/env python3
import sqlite3
import time
import torch
import os
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from web_scraper import WebScraper
from nlp_processor import NLPProcessor

class AnswerOptimizer:
    def __init__(self, db_path='answer_optimizer.db'):
        self.model_name = 'facebook/bart-large-cnn'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.db_path = db_path
        self._initialize_database()
        self._learn_from_database()
        self._running = True
        self.web_scraper = WebScraper()
        self.nlp_processor = NLPProcessor()

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

    def _learn_from_database(self):
        data = self.get_optimized_answers()
        print(f"Learning from {len(data)} previous optimizations to enhance future answers...")

    def optimize_answer(self, query: str, answer: str) -> str:
        input_length = len(self.tokenizer.encode(answer, add_special_tokens=False))
        max_length = min(150, input_length)
        min_length = min(30, max_length - 1)
        optimized = self.summarizer(answer, max_length=max_length, min_length=min_length, do_sample=False)
        optimized_answer = optimized[0]['summary_text']
        sentiment = self.analyze_sentiment(answer)
        self._store_data(query, answer, optimized_answer, sentiment)
        return optimized_answer

    def analyze_sentiment(self, text: str) -> str:
        analysis = self.sentiment_analyzer(text)
        return analysis[0]['label']

    def get_optimized_answers(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM Answers')
            return cursor.fetchall()

    def display_metrics(self, original: str, optimized: str):
        metrics = {
            "Original Length": len(original),
            "Optimized Length": len(optimized),
            "Compression Ratio": f"{len(optimized) / len(original) if len(original) > 0 else 0:.2%}",
        }
        print("\n--- Optimization Metrics ---")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("\n--- Copy and Paste the following metrics ---")
        print(f"Original Length: {metrics['Original Length']}")
        print(f"Optimized Length: {metrics['Optimized Length']}")
        print(f"Compression Ratio: {metrics['Compression Ratio']}")
        print("-----------------------------")

    def train(self, query: str, optimized_answer: str):
        # This is a placeholder for the actual training logic
        print(f"Training the model with query: {query}")
        print(f"Optimized answer: {optimized_answer}")
        # Here you would typically update your model based on this new data
        # For now, we'll just store it in the database
        self._store_data(query, "", optimized_answer, self.analyze_sentiment(optimized_answer))

def process_query(query, optimizer, num_results=5):
    scraped_data, _ = optimizer.web_scraper.scrape(query, num_results)
    nlp_response = optimizer.nlp_processor.process_nlp(scraped_data, query)
    optimized_answer = optimizer.optimize_answer(query, nlp_response['response'])
    return optimized_answer

def process_queries(queries, optimizer, num_results=5):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_query = {executor.submit(process_query, query, optimizer, num_results): query for query in queries}
        for future in tqdm(as_completed(future_to_query), total=len(future_to_query), desc="Processing queries"):
            query = future_to_query[future]
            try:
                optimized_answer = future.result()
                print(f"Query: {query}")
                print(f"Optimized Answer: {optimized_answer}")
                optimizer.display_metrics(query, optimized_answer)
            except Exception as exc:
                print(f"Query processing generated an exception: {exc}")

def read_topics_from_file(filename):
    with open(filename, 'r') as file:
        topics = file.readlines()
    return [topic.strip() for topic in topics if topic.strip()]

def main():
    parser = argparse.ArgumentParser(description="Optimize the quality of answers.")
    parser.add_argument('--queries', '-q', nargs='+', help='List of queries to optimize')
    parser.add_argument('--iterations', '-i', type=int, default=1, help='Number of batch iterations to perform (use -1 for infinite)')
    parser.add_argument('--topics-file', '-f', type=str, default='topics.txt', help='File containing topics to optimize')
    parser.add_argument('--train', action='store_true', help='Train the model using the provided topics')
    parser.add_argument('--num-results', '-n', type=int, default=5, help='Number of search results to retrieve for each query')
    args = parser.parse_args()

    queries = args.queries
    if queries is None:
        queries = read_topics_from_file(args.topics_file)

    batch_size = 3  # Adjust batch size as needed
    iterations = args.iterations
    num_results = args.num_results

    optimizer = AnswerOptimizer()

    if iterations == -1:
        print("Running indefinitely. Press 'q' and Enter to stop...")
        with tqdm(total=None, desc="Training progress") as pbar:
            while optimizer._running:
                for i in range(0, len(queries), batch_size):
                    print(f"\nProcessing batch {i // batch_size + 1}...\n", flush=True)
                    process_queries(queries[i:i + batch_size], optimizer, num_results)
                    print("\n", flush=True)  # Add space between batches