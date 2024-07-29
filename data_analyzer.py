#!/usr/bin/env python3
import logging
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from database_handler import DatabaseHandler
import warnings
import json
import sys
import torch
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DataAnalyzer:
    def __init__(self, db_path, device=None):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_handler = DatabaseHandler(db_path)
        self.start_time = 0
        self.end_time = 0
        self.original_texts = []
        self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        model_name = "sshleifer/distilbart-cnn-12-6"
        model_path = self.download_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.summarizer = pipeline("summarization", model=model_path, device=0 if torch.cuda.is_available() else -1)
        print(f"DataAnalyzer using device: {self.device}")

    def download_model(self, model_name, cache_dir='cached_models'):
        model_path = os.path.join(cache_dir, model_name.replace('/', '_'))
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_path).save_pretrained(model_path)
            AutoTokenizer.from_pretrained(model_name, cache_dir=model_path).save_pretrained(model_path)
        return model_path

    def analyze_data(self, data, query):
        try:
            documents = [item['content'] for item in data] + [query]
            X = self.vectorizer.fit_transform(documents)
            kmeans = KMeans(n_clusters=min(5, len(documents)), random_state=0).fit(X)
            labels = kmeans.labels_
            
            feature_names = self.vectorizer.get_feature_names_out()
            key_terms = self._extract_key_terms(X, feature_names)
            
            # Incorporate previously learned key terms
            previous_terms = self.db_handler.get_key_terms(query)
            if previous_terms:
                key_terms = list(set(key_terms + previous_terms))
            
            return labels, key_terms
        except ValueError as e:
            logger.error(f"Error during analysis: {e}")
            return None, None

    def _extract_key_terms(self, X, feature_names, top_n=10):
        importance = X.sum(axis=0).A1
        top_indices = importance.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]

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
            
            word_freq = self._analyze_word_frequency(summaries + [query])
            self.db_handler.save_word_frequency(query, word_freq)
            
            return summaries
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return None

    def _analyze_word_frequency(self, texts):
        words = ' '.join(texts).lower().split()
        return Counter(words).most_common(20)

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

        self.db_handler.save_metrics(query=query, num_results=len(self.original_texts), 
                                     total_text_length=total_text_length, 
                                     total_summary_length=total_summary_length, 
                                     processing_time=processing_time, 
                                     metric_name='', metric_value=0)

    def learn_from_answer(self, query, answer):
        self.db_handler.store_data({'query': query, 'response': answer})
        self.vectorizer.fit_transform([query, answer])
        _, key_terms = self.analyze_data([{'content': answer}], query)
        self.db_handler.save_key_terms(query, key_terms)
        
        # Learn from semantic similarity
        answer_embedding = self.sentence_transformer.encode(answer)
        self.db_handler.store_embedding(query, answer_embedding)

    def get_similar_answers(self, query, top_k=3):
        query_embedding = self.sentence_transformer.encode(query)
        similar_queries = self.db_handler.get_similar_embeddings(query_embedding, top_k)
        return [self.db_handler.retrieve_data(q) for q in similar_queries]

    def process_query(self, query, data):
        self.start_time = time.time()
        labels, key_terms = self.analyze_data(data, query)
        summaries = self.summarize_content(data, query)

        if labels is not None and summaries:
            logger.info(f"Labels: {labels}")
            logger.info(f"Key Terms: {key_terms}")
            
            # Incorporate similar previous answers
            similar_answers = self.get_similar_answers(query)
            combined_summaries = summaries + similar_answers
            
            final_answer = ' '.join(combined_summaries)
            self.end_time = time.time()
            self.display_final_answer(final_answer, query)
            self.learn_from_answer(query, final_answer)
            return final_answer
        else:
            logger.error("Failed to process query")
            return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_analyzer.py <data_file> <query>")
        sys.exit(1)

    data_file = sys.argv[1]
    query = sys.argv[2]
    with open(data_file, 'r') as f:
        data = json.load(f)

    analyzer = DataAnalyzer("ai_assistant.db")
    result = analyzer.process_query(query, data)
    if result:
        print(f"Final Answer: {result}")
    else:
        print("Failed to generate an answer.")