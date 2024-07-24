#!/usr/bin/env python

import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from advanced_nlp_processor import AdvancedNLPProcessor
from web_scraper import WebScraper
from fact_checker import FactChecker
from data_analyzer import DataAnalyzer
from database_handler import DatabaseHandler

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
limiter = Limiter(app, key_func=get_remote_address)

db_handler = DatabaseHandler()
scraper = WebScraper()
fact_checker = FactChecker(db_handler)
data_analyzer = DataAnalyzer('ai_assistant.db')
nlp_processor = AdvancedNLPProcessor()

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
            return jsonify({'query': user_query, 'response': cached_response})
        
        # Scrape web data
        logging.info("Starting web scraping...")
        scraped_data = scraper.scrape(user_query)
        
        # Analyze data
        logging.info("Analyzing data...")
        analyzed_data = data_analyzer.analyze_data(scraped_data)
        
        # Fact-check data
        logging.info("Fact-checking data...")
        checked_data = fact_checker.check(analyzed_data)
        
        # Process NLP
        logging.info("Processing NLP...")
        texts = [item['content'] for item in checked_data if isinstance(item.get('content'), str)]
        result = nlp_processor.process(user_query, texts)
        
        # Store in database
        logging.info("Storing data in database...")
        db_handler.store_data({'query': user_query, 'response': result['answer']})
        
        return jsonify({
            'query': user_query,
            'response': result['answer'],
            'summary': result['summary'],
            'entities': result['entities'],
            'key_information': result['key_information']
        })
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def feedback():
    data = request.json
    if not data or 'query' not in data or 'rating' not in data:
        return jsonify({'error': 'Invalid feedback data'}), 400
    
    try:
        db_handler.store_feedback(data['query'], data['rating'], data.get('comment', ''))
        return jsonify({'message': 'Feedback stored successfully'})
    except Exception as e:
        logging.error(f"Error storing feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        stats = db_handler.get_feedback_statistics()
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error retrieving statistics: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
