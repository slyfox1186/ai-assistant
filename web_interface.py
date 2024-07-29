#!/usr/bin/env python3

import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from nlp_processor import NLPProcessor
from web_scraper import WebScraper
from fact_checker import FactChecker
from data_analyzer import DataAnalyzer
from database_handler import DatabaseHandler
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
limiter = Limiter(app, key_func=get_remote_address)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db_handler = DatabaseHandler(device=device)
scraper = WebScraper(device=device)
fact_checker = FactChecker(db_handler, device=device)
data_analyzer = DataAnalyzer("ai_assistant.db", device=device)
nlp_processor = NLPProcessor()

# Initialize a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def process_and_validate_answer(query, raw_answer):
    # Use the summarizer to get the key points
    summary = summarizer(raw_answer, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    
    # Use the NLP processor to ensure the answer is relevant to the query
    processed_answer = nlp_processor.ensure_relevance(query, summary)
    
    # Remove any non-relevant text from the processed answer
    cleaned_answer = clean_irrelevant_text(processed_answer)
    
    # Use the fact checker to validate the cleaned answer
    validated_answer = fact_checker.validate_answer(cleaned_answer)
    
    # Return the validated answer
    return validated_answer

@app.route('/query', methods=['POST'])
@limiter.limit("5 per minute")
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query']

    try:
        # Check cache first
        cached_response = db_handler.retrieve_data(user_query)
        if cached_response:
            logging.info("Found cached response.")
            validated_response = process_and_validate_answer(user_query, cached_response)
            return jsonify({'query': user_query, 'response': validated_response})

        # Scrape web data
        logging.info("Starting web scraping...")
        scraped_data, _ = scraper.scrape(user_query, 5)

        # Analyze data
        logging.info("Analyzing data...")
        analyzed_data = data_analyzer.analyze_data(scraped_data)

        # Process NLP
        logging.info("Processing NLP...")
        raw_response = nlp_processor.process_nlp(analyzed_data, user_query)

        # Process and validate the answer
        validated_response = process_and_validate_answer(user_query, raw_response)

        # Store in database
        logging.info("Storing data in database...")
        db_handler.store_data({'query': user_query, 'response': validated_response})

        return jsonify({'query': user_query, 'response': validated_response})
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def feedback():
    data = request.json
    if not data or 'query' not in data or 'response' not in data or 'rating' not in data:
        return jsonify({'error': 'Invalid feedback data'}), 400

    try:
        query = data['query']
        response = data['response']
        rating = data['rating']
        
        # Store feedback in the database
        db_handler.store_feedback(query, rating, data.get('comment', ''))
        
        return jsonify({'message': 'Feedback processed successfully'})
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        stats = db_handler.get_feedback_statistics()
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error retrieving statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
@limiter.limit("1 per hour")
def train():
    try:
        adaptive_learner.periodic_training(num_samples=100)
        return jsonify({'message': 'Periodic training completed successfully'})
    except Exception as e:
        logging.error(f"Error during periodic training: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
