#!/usr/bin/env python

import logging
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
from database_handler import DatabaseHandler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DataAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.summarizer = pipeline("summarization")
        self.db_handler = DatabaseHandler(db_path)
        self.start_time = 0
        self.end_time = 0
        self.original_texts = []

    def analyze_data(self, data):
        try:
            vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
            X = vectorizer.fit_transform([item['content'] for item in data])
            kmeans = KMeans(n_clusters=min(5, len(data)), random_state=0).fit(X)
            labels = kmeans.labels_
            return labels
        except ValueError as e:
            logger.error(f"Error during analysis: {e}")
            return None

    def summarize_content(self, data, query):
        try:
            summaries = []
            for item in data:
                summary = self.summarizer(item['content'], max_length=min(len(item['content']) // 2, 256), min_length=10, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            total_summary_length = sum(len(summary) for summary in summaries)
            self.db_handler.save_metrics(query=query, num_results=len(data), total_text_length=0, 
                                         total_summary_length=total_summary_length, processing_time=0, 
                                         metric_name='summary_length', metric_value=total_summary_length)
            
            return summaries
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return None

    def display_final_answer(self, final_response, query):
        print("\n--- Query Result ---")
        print(f"Query: {query}")
        print("\nSummary:")
        print(final_response)
        print("\n--- Data Metrics ---")
        total_summary_length = len(final_response)
        total_text_length = sum(len(text) for text in self.original_texts)
        processing_time = self.end_time - self.start_time
        print(f"Summary Length: {total_summary_length:,} characters")
        print(f"Original Text Length: {total_text_length:,} characters")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Compression Ratio: {total_summary_length / total_text_length:.2%}")
        print(f"Processing Speed: {total_text_length / processing_time:.2f} characters/second")
        print("---------------------")

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 3:
        print("Usage: python data_analyzer.py <data_file> <query>")
        sys.exit(1)

    data_file = sys.argv[1]
    query = sys.argv[2]
    with open(data_file, 'r') as f:
        data = json.load(f)

    analyzer = DataAnalyzer("ai_assistant.db")
    labels = analyzer.analyze_data(data)
    summaries = analyzer.summarize_content(data, query)

    if labels:
        logger.info(f"Labels: {labels}")
    if summaries:
        logger.info(f"Summaries: {summaries}")
        analyzer.display_final_answer(' '.join(summaries), query)