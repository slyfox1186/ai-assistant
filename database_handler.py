#!/usr/bin/env python

import sqlite3
import logging
from contextlib import contextmanager
import time
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DatabaseHandler:
    def __init__(self, db_name='ai_assistant.db'):
        self.db_name = db_name
        self.initialize_database()
    
    def initialize_database(self):
        db_exists = os.path.exists(self.db_name)
        
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            
            if not db_exists:
                logger.info(f"Creating new database: {self.db_name}")
                cursor.execute('''CREATE TABLE Data 
                                  (id INTEGER PRIMARY KEY, query TEXT, response TEXT, timestamp REAL)''')
                cursor.execute('''CREATE TABLE Feedback
                                  (id INTEGER PRIMARY KEY, query_id INTEGER, rating INTEGER, 
                                   comment TEXT, timestamp REAL,
                                   FOREIGN KEY(query_id) REFERENCES Data(id))''')
                cursor.execute('''CREATE TABLE metrics (
                                  id INTEGER PRIMARY KEY,
                                  metric_name TEXT NOT NULL,
                                  metric_value REAL NOT NULL)''')
            else:
                cursor.execute("PRAGMA table_info(Data)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'timestamp' not in columns:
                    logger.info("Adding timestamp column to Data table")
                    cursor.execute("ALTER TABLE Data ADD COLUMN timestamp REAL")
                cursor.execute("PRAGMA table_info(metrics)")
                metrics_columns = [column[1] for column in cursor.fetchall()]
                if not metrics_columns:
                    logger.info("Creating metrics table")
                    cursor.execute('''CREATE TABLE metrics (
                                      id INTEGER PRIMARY KEY,
                                      metric_name TEXT NOT NULL,
                                      metric_value REAL NOT NULL)''')
            
            conn.commit()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
        finally:
            conn.close()
    
    def store_data(self, data):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO Data (query, response, timestamp) VALUES (?, ?, ?)',
                           (data['query'], data['response'], time.time()))
            return cursor.lastrowid
    
    def retrieve_data(self, query):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT response FROM Data WHERE query=? ORDER BY timestamp DESC LIMIT 1', (query,))
            result = cursor.fetchone()
            if result:
                try:
                    parsed_response = eval(result[0])
                    if isinstance(parsed_response, dict) and 'response' in parsed_response:
                        return result[0]
                except:
                    logger.warning(f"Invalid cached response for query: {query}")
                    self.clear_cache(query)
            return None
    
    def store_feedback(self, query, rating, comment):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM Data WHERE query=? ORDER BY timestamp DESC LIMIT 1', (query,))
            query_id = cursor.fetchone()
            if query_id:
                cursor.execute('INSERT INTO Feedback (query_id, rating, comment, timestamp) VALUES (?, ?, ?, ?)',
                               (query_id[0], rating, comment, time.time()))
            else:
                logger.warning(f"No matching query found for feedback: {query}")
    
    def clear_cache(self, query):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM Data WHERE query=?', (query,))
            conn.commit()
            logger.info(f"Cleared cache for query: {query}")
    
    def clean_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''DELETE FROM Data WHERE id NOT IN 
                              (SELECT id FROM Data GROUP BY query HAVING MAX(timestamp))''')
            conn.commit()
            logger.info(f"Cleaned {cursor.rowcount} outdated entries from the database.")
    
    def get_feedback_statistics(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT AVG(rating), COUNT(*) FROM Feedback''')
            avg_rating, count = cursor.fetchone()
            return {'average_rating': avg_rating, 'total_feedback': count}
    
    def save_metrics(self, query, num_results, total_text_length, total_summary_length, processing_time, metric_name, metric_value):
        logger.info(f"Saving metrics for query: {query}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics (metric_name, metric_value)
                VALUES (?, ?)
            ''', (metric_name, metric_value))
            conn.commit()