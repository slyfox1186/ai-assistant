#!/usr/bin/env python

import argparse
import asyncio
import time
import os
from transformers import pipeline, AutoTokenizer
import torch
from database_handler import DatabaseHandler
from web_scraper import WebScraper
from data_analyzer import DataAnalyzer
from nlp_processor import NLPProcessor
from answer_enhancer import AnswerEnhancer
from answer_optimizer import AnswerOptimizer
import warnings
import transformers
import logging

# Set environment variable to fix MKL error
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# Custom logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_cleaned_text(text):
    print(text)

async def main():
    parser = argparse.ArgumentParser(description='Ask a question and get an answer by searching the web.')
    parser.add_argument('question', type=str, help='The question you want to ask.')
    parser.add_argument('--results', type=int, default=5, help='Number of search results to retrieve.')
    parser.add_argument('--db-name', type=str, default='ai_assistant.db', help='Database name')
    parser.add_argument('--iterations', '-i', type=int, default=1, help='Number of batch iterations to perform for optimization')
    parser.add_argument('--train', action='store_true', help='Train the model using the provided topics')
    args = parser.parse_args()

    query = args.question
    num_results = args.results
    db_name = args.db_name
    iterations = args.iterations
    train = args.train

    logger.info(f"Processing query: {query}")
    logger.info(f"Number of results: {num_results}")
    logger.info(f"Database name: {db_name}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Train flag: {train}")

    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

    db_handler = DatabaseHandler(db_name)
    web_scraper = WebScraper()
    data_analyzer = DataAnalyzer(db_name)
    nlp_processor = NLPProcessor()
    answer_enhancer = AnswerEnhancer()
    answer_optimizer = AnswerOptimizer()

    cached_answer = db_handler.retrieve_data(query)
    if cached_answer:
        logger.info("Using cached answer")
        print_cleaned_text(cached_answer)
        return

    try:
        start_time = time.time()
        logger.info("Starting web scraping")
        scraped_data, formatted_data = web_scraper.scrape(query, num_results)

        if not scraped_data:
            logger.warning("No relevant information found")
            print_cleaned_text("I'm sorry, but I couldn't find any relevant information for your question.")
            return

        logger.debug("Scraped data:")
        for item in formatted_data:
            logger.debug(item)

        data_analyzer.start_time = start_time
        data_analyzer.original_texts = [item['content'] for item in scraped_data if isinstance(item.get('content'), str)]

        logger.info("Processing NLP")
        try:
            processed_response = nlp_processor.process_nlp(scraped_data, query)
        except Exception as e:
            logger.exception(f"Error during NLP processing: {e}")
            print_cleaned_text("An error occurred during NLP processing. Please try again.")
            return

        if not isinstance(processed_response, dict) or 'response' not in processed_response:
            logger.error(f"Unexpected response format from NLP processor: {processed_response}")
            print_cleaned_text("Received an unexpected response format from the NLP processor. Please try again.")
            return

        logger.info("Enhancing answer")
        try:
            enhanced_response = answer_enhancer.enhance_answer(query, processed_response['response'])
        except Exception as e:
            logger.error(f"Error during answer enhancement: {e}")
            enhanced_response = processed_response['response']  # Use the original response if enhancement fails

        logger.info("Optimizing answer")
        try:
            optimized_response = answer_optimizer.optimize_answer(query, enhanced_response)
        except Exception as e:
            logger.error(f"Error during answer optimization: {e}")
            optimized_response = enhanced_response  # Use the enhanced response if optimization fails

        data_analyzer.end_time = time.time()
        
        logger.info("Displaying final answer")
        data_analyzer.display_final_answer(optimized_response, query)
        
        total_summary_length = len(optimized_response)
        total_text_length = sum(len(t) for t in data_analyzer.original_texts if isinstance(t, str))
        processing_time = data_analyzer.end_time - data_analyzer.start_time
        
        logger.info(f"Storing data in database")
        db_handler.store_data({'query': query, 'response': optimized_response})
        db_handler.save_metrics(query=query, num_results=num_results, total_text_length=total_text_length, 
                                total_summary_length=total_summary_length, processing_time=processing_time, 
                                metric_name='', metric_value=0)
        
        if train:
            logger.info("Training the model using the optimized answer.")
            answer_optimizer.train(query, optimized_response)

    except Exception as e:
        logger.exception(f"An error occurred while processing the request: {e}")
        print_cleaned_text(f"An error occurred while processing your request. Please check the logs for more details. Error: {str(e)}")
        enhanced_response = None
        optimized_response = None

    if optimized_response:
        print_cleaned_text(optimized_response)
    elif enhanced_response:
        print_cleaned_text(enhanced_response)
    else:
        print_cleaned_text("Sorry, I couldn't generate a response. Please try again.")
        
if __name__ == "__main__":
    asyncio.run(main())