#!/usr/bin/env python3

import logging
import sqlite3
from contextlib import contextmanager
import time
import os
import torch
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from regex_cleaner import RegexCleaner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DatabaseHandler:
    def __init__(self, db_name='ai_assistant.db', device=None):
        self.db_name = db_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_database()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.exclusion_files = self.load_exclusion_files()
        self.regex_cleaner = RegexCleaner(exclusion_files=self.exclusion_files)
        logger.info(f"DatabaseHandler using device: {self.device}")

    def load_exclusion_files(self, directory='training_data'):
        exclusion_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                try:
    
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        exclusion_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error loading exclusion file {file_path}: {e}")
        return exclusion_files

    def initialize_database(self):
        db_exists = os.path.exists(self.db_name)

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()

            if not db_exists:
                logger.info(f"Creating new database: {self.db_name}")
                self._create_tables(cursor)
            else:
                logger.info(f"Database {self.db_name} exists. Checking and updating tables...")
                self._update_tables(cursor)

            conn.commit()

    def _create_tables(self, cursor):
        cursor.execute('''CREATE TABLE IF NOT EXISTS Data
                          (id INTEGER PRIMARY KEY, query TEXT, response TEXT, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS Feedback
                          (id INTEGER PRIMARY KEY, query_id INTEGER, rating INTEGER, comment TEXT, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metrics
                          (id INTEGER PRIMARY KEY, query TEXT, num_results INTEGER, total_text_length INTEGER,
                           total_summary_length INTEGER, processing_time REAL, metric_name TEXT, metric_value REAL, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS key_terms
                          (id INTEGER PRIMARY KEY, query TEXT, terms TEXT, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS irrelevant_text
                          (id INTEGER PRIMARY KEY, query TEXT, irrelevant_content TEXT, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings
                          (id INTEGER PRIMARY KEY, query TEXT, embedding BLOB, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS model_data
                          (id INTEGER PRIMARY KEY, query TEXT, response TEXT, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS word_frequency
                          (id INTEGER PRIMARY KEY, query TEXT, word TEXT, frequency INTEGER, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS irrelevant_phrases
                          (id INTEGER PRIMARY KEY, phrase TEXT, timestamp REAL)''')

    def _update_tables(self, cursor):
        tables = self._get_existing_tables(cursor)
        
        if 'Data' in tables:
            self._add_column_if_missing(cursor, 'Data', 'timestamp', 'REAL')
        if 'Feedback' in tables:
            self._add_column_if_missing(cursor, 'Feedback', 'timestamp', 'REAL')
        if 'metrics' in tables:
            self._add_column_if_missing(cursor, 'metrics', 'timestamp', 'REAL')
        if 'key_terms' in tables:
            self._add_column_if_missing(cursor, 'key_terms', 'timestamp', 'REAL')
        if 'irrelevant_text' not in tables:
            self._create_tables(cursor)
        if 'embeddings' not in tables:
            self._create_tables(cursor)
        if 'model_data' not in tables:
            self._create_tables(cursor)
        if 'word_frequency' not in tables:
            self._create_tables(cursor)
        if 'irrelevant_phrases' not in tables:
            self._create_tables(cursor)

    def _get_existing_tables(self, cursor):
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return {row[0] for row in cursor.fetchall()}

    def _add_column_if_missing(self, cursor, table_name, column_name, column_type):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        if column_name not in columns:
            logger.info(f"Adding {column_name} column to {table_name} table")
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
        finally:
            conn.close()

    def store_data(self, data):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cleaned_query = self.regex_cleaner.clean_text(data['query'])
                cleaned_response = self.regex_cleaner.clean_text(data['response'])
                logger.info(f"Storing data: query={cleaned_query}, response={cleaned_response}")
                cursor.execute('INSERT INTO Data (query, response, timestamp) VALUES (?, ?, ?)',
                               (cleaned_query, cleaned_response, time.time()))
                conn.commit()
            logger.info(f"Stored data for query: {data['query']}")
        except sqlite3.Error as e:
            logger.error(f"Error storing data in the database: {e}")
            raise

    def retrieve_data(self, query):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT response FROM Data WHERE query=? ORDER BY timestamp DESC LIMIT 1', (cleaned_query,))
            result = cursor.fetchone()
            if result:
                return self.regex_cleaner.clean_text(result[0])
            return None

    def store_feedback(self, query, original_answer, enhanced_answer, user_feedback):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO Feedback (query_id, rating, comment, timestamp) VALUES (?, ?, ?, ?)',
                           (query, user_feedback, f"Original: {original_answer}\nEnhanced: {enhanced_answer}", time.time()))
            conn.commit()

    def store_irrelevant_text(self, query, irrelevant_content):
        cleaned_query = self.regex_cleaner.clean_text(query)
        cleaned_irrelevant_content = self.regex_cleaner.clean_text(irrelevant_content)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO irrelevant_text (query, irrelevant_content, timestamp) VALUES (?, ?, ?)',
                           (cleaned_query, cleaned_irrelevant_content, time.time()))
            conn.commit()

    def get_similar_queries(self, query, limit=5):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT query, response FROM Data ORDER BY timestamp DESC LIMIT 100')
            recent_queries = cursor.fetchall()
            
            if not recent_queries:
                return []
            
            queries, responses = zip(*recent_queries)
            self.vectorizer.fit(queries)
            query_vector = self.vectorizer.transform([cleaned_query])
            all_vectors = self.vectorizer.transform(queries)
            
            similarities = cosine_similarity(query_vector, all_vectors).flatten()
            most_similar_indices = similarities.argsort()[-limit:][::-1]
            
            return [{'query': queries[i], 'response': responses[i]} for i in most_similar_indices]

    def save_metrics(self, query, num_results, total_text_length, total_summary_length, processing_time, metric_name, metric_value):
        try:
            cleaned_query = self.regex_cleaner.clean_text(query)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                logger.info(f"Saving metrics for query: {cleaned_query}")
                cursor.execute('''
                    INSERT INTO metrics (query, num_results, total_text_length, total_summary_length, processing_time, metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (cleaned_query, num_results, total_text_length, total_summary_length, processing_time, metric_name, metric_value, time.time()))
                conn.commit()
            logger.info(f"Saved metrics for query: {query}")
        except sqlite3.Error as e:
            logger.error(f"Error saving metrics to the database: {e}")
            raise

    def save_key_terms(self, query, terms):
        try:
            cleaned_query = self.regex_cleaner.clean_text(query)
            cleaned_terms = self.regex_cleaner.clean_text(','.join(terms))
            with self.get_connection() as conn:
                cursor = conn.cursor()
                logger.info(f"Saving key terms for query: {cleaned_query}")
                cursor.execute('INSERT INTO key_terms (query, terms, timestamp) VALUES (?, ?, ?)',
                               (cleaned_query, cleaned_terms, time.time()))
                conn.commit()
            logger.info(f"Saved key terms for query: {query}")
        except sqlite3.Error as e:
            logger.error(f"Error saving key terms to the database: {e}")
            raise

    def get_key_terms(self, query):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT terms FROM key_terms WHERE query=? ORDER BY timestamp DESC LIMIT 1', (cleaned_query,))
            result = cursor.fetchone()
            if result:
                return self.regex_cleaner.clean_text(result[0]).split(',')
            return []

    def get_feedback_statistics(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT AVG(rating), COUNT(*) FROM Feedback''')
            avg_rating, count = cursor.fetchone()
            return {'average_rating': avg_rating, 'total_feedback': count}

    def clear_cache(self, query):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Data WHERE query = ?", (cleaned_query,))
            conn.commit()
        logger.info(f"Cleared cache for query: {query}")

    def store_embedding(self, query, embedding):
        try:
            cleaned_query = self.regex_cleaner.clean_text(query)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                logger.info(f"Storing embedding for query: {cleaned_query}")
                cursor.execute('INSERT INTO embeddings (query, embedding, timestamp) VALUES (?, ?, ?)',
                               (cleaned_query, embedding.tobytes(), time.time()))
                conn.commit()
            logger.info(f"Stored embedding for query: {query}")
        except sqlite3.Error as e:
            logger.error(f"Error storing embedding in the database: {e}")
            raise

    def get_similar_embeddings(self, query_embedding, top_k=3):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT query, embedding FROM embeddings ORDER BY timestamp DESC LIMIT 100')
            results = cursor.fetchall()
            
            if not results:
                return []
            
            queries, embeddings = zip(*results)
            embeddings = [np.frombuffer(emb) for emb in embeddings]
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            most_similar_indices = similarities.argsort()[-top_k:][::-1]
            
            return [queries[i] for i in most_similar_indices]

    def save_word_frequency(self, query, word_freq):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for word, freq in word_freq:
                cleaned_word = self.regex_cleaner.clean_text(word)
                logger.info(f"Saving word frequency: {cleaned_word} -> {freq}")
                cursor.execute('''
                    INSERT INTO word_frequency (query, word, frequency, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (cleaned_query, cleaned_word, freq, time.time()))
            conn.commit()
            logger.info(f"Saved word frequency for query: {query}")

    def store_model_data(self, query, response):
        try:
            cleaned_query = self.regex_cleaner.clean_text(query)
            cleaned_response = self.regex_cleaner.clean_text(response)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                logger.info(f"Storing model data: query={cleaned_query}, response={cleaned_response}")
                cursor.execute('INSERT INTO model_data (query, response, timestamp) VALUES (?, ?, ?)',
                               (cleaned_query, cleaned_response, time.time()))
                conn.commit()
            logger.info(f"Stored model data for query: {query}")
        except sqlite3.Error as e:
            logger.error(f"Error storing model data in the database: {e}")
            raise

    def retrieve_model_data(self, query):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT response FROM model_data WHERE query=? ORDER BY timestamp DESC LIMIT 1', (cleaned_query,))
            result = cursor.fetchone()
            if result:
                return self.regex_cleaner.clean_text(result[0])
            return None

    def get_irrelevant_phrases(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT phrase FROM irrelevant_phrases')
            results = cursor.fetchall()
            return [result[0] for result in results]

    def update_irrelevant_phrases(self, phrases):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM irrelevant_phrases')
            for phrase in phrases:
                cursor.execute('INSERT INTO irrelevant_phrases (phrase, timestamp) VALUES (?, ?)',
                               (phrase, time.time()))
            conn.commit()
        logger.info(f"Updated irrelevant phrases: {len(phrases)} phrases stored")

    def store_enhanced_answer(self, query, original_answer, enhanced_answer):
        try:
            cleaned_query = self.regex_cleaner.clean_text(query)
            cleaned_original = self.regex_cleaner.clean_text(original_answer)
            cleaned_enhanced = self.regex_cleaner.clean_text(enhanced_answer)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                logger.info(f"Storing enhanced answer for query: {cleaned_query}")
                cursor.execute('''
                    INSERT INTO model_data (query, response, timestamp)
                    VALUES (?, ?, ?)
                ''', (cleaned_query, f"Original: {cleaned_original}\nEnhanced: {cleaned_enhanced}", time.time()))
                conn.commit()
            logger.info(f"Stored enhanced answer for query: {query}")
        except sqlite3.Error as e:
            logger.error(f"Error storing enhanced answer in the database: {e}")
            raise

    def get_relevant_info(self, query):
        cleaned_query = self.regex_cleaner.clean_text(query)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT response FROM model_data 
                WHERE query LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (f"%{cleaned_query}%",))
            result = cursor.fetchone()
            if result:
                return self.regex_cleaner.clean_text(result[0])
            return None

if __name__ == "__main__":
    db_handler = DatabaseHandler()
    # Example usage
    db_handler.store_data({'query': 'Example query', 'response': 'Example response'})
    result = db_handler.retrieve_data('Example query')
    print(f"Retrieved data: {result}")
