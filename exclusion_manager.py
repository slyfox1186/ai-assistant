#!/usr/bin/env python3

import os
import json
import sqlite3
import logging
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import argparse
from web_scraper import WebScraper

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('exclusion_manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Database setup
DB_NAME = 'exclusion_phrases.db'

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS exclusion_phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase TEXT UNIQUE NOT NULL)''')
    conn.commit()
    conn.close()

def add_phrases_to_db(phrases):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for phrase in phrases:
        try:
            c.execute("INSERT INTO exclusion_phrases (phrase) VALUES (?)", (phrase,))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()

def get_all_phrases_from_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT phrase FROM exclusion_phrases")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def save_phrases_to_json(phrases, json_path='exclusion_criteria.json'):
    data = {"phrases_to_exclude": phrases}
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_existing_phrases(json_path='exclusion_criteria.json'):
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            return data.get("phrases_to_exclude", [])
    return []

def update_exclusion_phrases(scraped_data, min_freq=5):
    all_text = " ".join([item['content'] for item in scraped_data])
    tokens = word_tokenize(all_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    freq_dist = FreqDist(filtered_tokens)
    
    common_phrases = [phrase for phrase, freq in freq_dist.items() if freq >= min_freq]
    existing_phrases = get_all_phrases_from_db()
    
    new_phrases = set(common_phrases) - set(existing_phrases)
    add_phrases_to_db(new_phrases)
    
    updated_phrases = get_all_phrases_from_db()
    save_phrases_to_json(updated_phrases)
    
    logger.info(f"Updated exclusion phrases. New phrases added: {len(new_phrases)}")
    return updated_phrases

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Update exclusion phrases based on scraped data.')
    parser.add_argument('query', type=str, help='The query to search for.')
    parser.add_argument('--results', type=int, default=5, help='Number of search results to retrieve.')
    args = parser.parse_args()

    # Initialize database
    initialize_db()
    
    # Load existing phrases from JSON file (if any)
    existing_phrases = load_existing_phrases()
    
    # Add existing phrases to the database
    add_phrases_to_db(existing_phrases)
    
    # Use web scraper to get scraped data
    scraper = WebScraper()
    scraped_data, _ = scraper.scrape(args.query, args.results)
    
    # Update exclusion phrases based on scraped data
    updated_phrases = update_exclusion_phrases(scraped_data)
    print(f"Updated exclusion phrases: {updated_phrases}")

if __name__ == "__main__":
    main()
