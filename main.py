#!/usr/bin/env python3

import os
import logging
import argparse
import asyncio
import time
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from database_handler import DatabaseHandler
from web_scraper import WebScraper
from data_analyzer import DataAnalyzer
from nlp_processor import NLPProcessor
from answer_enhancer import AnswerEnhancer
from answer_optimizer import AnswerOptimizer
from adaptive_learning import AdaptiveLearning
from fact_checker import FactChecker
from regex_cleaner import RegexCleaner
from grammar_enhancer import GrammarEnhancer
import warnings
import transformers
from diskcache import Cache
import concurrent.futures

# Set environment variables to debug CUDA issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# Initialize cache
cache = Cache('cache_dir')

def print_cleaned_text(text):
    print("\n--- Query Result ---")
    print(f"\nSummary:\n{text}")
    print("\n--- End of Result ---\n")

def download_model(model_name, cache_dir='model_cache'):
    model_path = os.path.join(cache_dir, model_name.replace('/', '_'))
    if not os.path.exists(model_path):
        logger.info(f"Downloading model {model_name} to cache")
        os.makedirs(model_path, exist_ok=True)
        try:
            AutoModelForSeq2SeqLM.from_pretrained(model_name).save_pretrained(model_path)
            AutoTokenizer.from_pretrained(model_name).save_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    else:
        logger.info(f"Using cached model {model_name}")
    return model_path

async def main():
    parser = argparse.ArgumentParser(description='Ask a question and get an answer by searching the web.')
    parser.add_argument('question', type=str, nargs='?', help='The question you want to ask.')
    parser.add_argument('--results', type=int, default=5, help='Number of search results to retrieve.')
    parser.add_argument('--db-name', type=str, default='ai_assistant.db', help='Database name')
    parser.add_argument('--iterations', '-i', type=int, default=1, help='Number of batch iterations for optimization')
    parser.add_argument('--train', action='store_true', help='Train the model using the provided topics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--arg1', type=str, help='First argument to modify the query processing')
    parser.add_argument('--arg2', type=str, help='Second argument to modify the query processing')
    args = parser.parse_args()

    if not args.question:
        parser.error("the following arguments are required: question")

    query = args.question
    num_results = args.results
    db_name = args.db_name
    iterations = args.iterations
    train = args.train
    verbose = args.verbose
    arg1 = args.arg1
    arg2 = args.arg2

    logger.info(f"Processing query: {query}")

    try:
        db_handler = DatabaseHandler(db_name)
        regex_cleaner = RegexCleaner(exclusion_files=db_handler.exclusion_files)
    except Exception as e:
        logger.error(f"Error initializing database or RegexCleaner: {e}")
        print_cleaned_text("An error occurred while initializing the database or RegexCleaner. Please check the logs for more details.")
        return

    web_scraper = WebScraper()
    data_analyzer = DataAnalyzer(db_name)
    nlp_processor = NLPProcessor()
    answer_enhancer = AnswerEnhancer()
    answer_optimizer = AnswerOptimizer()
    adaptive_learner = AdaptiveLearning()
    fact_checker = FactChecker(db_handler)
    grammar_enhancer = GrammarEnhancer(db_name)

    try:
        adaptive_learner.load_latest_model()
    except Exception as e:
        logger.error(f"Error loading adaptive learning model: {e}")
        print_cleaned_text("An error occurred while loading the adaptive learning model. Please check the logs for more details.")
        return

    model_path = download_model("sshleifer/distilbart-cnn-12-6")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    summarizer = pipeline("summarization", model=model_path, tokenizer=tokenizer, device=-1)

    cached_answer = cache.get(query)
    new_answer = None

    try:
        start_time = time.time()
        logger.info("Starting web scraping")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(web_scraper.scrape, query, num_results)
            scraped_data, formatted_data = future.result()

        if verbose:
            print("\n--- Scraped Data ---")
            for data in scraped_data:
                print(data['content'])

        if not scraped_data:
            logger.warning("No relevant information found")
            print_cleaned_text("I'm sorry, but I couldn't find any relevant information for your question.")
            return

        data_analyzer.start_time = start_time
        data_analyzer.original_texts = [item['content'] for item in scraped_data if isinstance(item.get('content'), str)]

        logger.info("Processing NLP")
        processed_response = nlp_processor.process_nlp(scraped_data, query, arg1, arg2)

        if verbose:
            print("\n--- Processed NLP Response ---")
            print(processed_response)

        if not isinstance(processed_response, dict) or 'response' not in processed_response:
            logger.error(f"Unexpected response format from NLP processor: {processed_response}")
            print_cleaned_text("Received an unexpected response format from the NLP processor. Please try again.")
            return

        logger.info("Enhancing answer")
        enhanced_response = answer_enhancer.enhance_answer(query, processed_response['response'])
        if verbose:
            print("\n--- Enhanced Response ---")
            print(enhanced_response)

        logger.info("Optimizing answer")
        optimized_response = answer_optimizer.optimize_answer(query, enhanced_response)
        if verbose:
            print("\n--- Optimized Response ---")
            print(optimized_response)

        logger.info("Applying adaptive learning")
        adaptive_response = adaptive_learner.adaptive_response(
            query=query,
            max_length=1024,
            max_new_tokens=50
        )
        if verbose:
            print("\n--- Adaptive Response ---")
            print(adaptive_response)

        logger.info("Fact-checking the response")
        fact_checked_data = fact_checker.check(query, adaptive_response)
        if isinstance(fact_checked_data, tuple):
            is_misinformation, fact_check_details = fact_checked_data
            if not is_misinformation:
                new_answer = adaptive_response
            else:
                new_answer = adaptive_response + "\n\nNote: This information may require further verification."
        else:
            new_answer = adaptive_response + "\n\nNote: Unable to verify the information."
        if verbose:
            print("\n--- Fact-checked Response ---")
            print(new_answer)

        logger.info("Enhancing grammar")
        new_answer = grammar_enhancer.enhance_grammar(query, new_answer)

        new_answer = regex_cleaner.clean_text(new_answer.strip())

        # If the new answer is the same as the cached answer, rescrape and reprocess
        if new_answer == cached_answer:
            logger.info("New answer is the same as cached answer. Re-scraping data.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(web_scraper.scrape, query, num_results)
                scraped_data, formatted_data = future.result()

            if verbose:
                print("\n--- Scraped Data After Re-scraping ---")
                for data in scraped_data:
                    print(data['content'])

            data_analyzer.original_texts = [item['content'] for item in scraped_data if isinstance(item.get('content'), str)]

            processed_response = nlp_processor.process_nlp(scraped_data, query, arg1, arg2)

            logger.info("Enhancing answer after re-scraping")
            enhanced_response = answer_enhancer.enhance_answer(query, processed_response['response'])
            if verbose:
                print("\n--- Enhanced Response After Re-scraping ---")
                print(enhanced_response)

            logger.info("Optimizing answer after re-scraping")
            optimized_response = answer_optimizer.optimize_answer(query, enhanced_response)
            if verbose:
                print("\n--- Optimized Response After Re-scraping ---")
                print(optimized_response)

            logger.info("Applying adaptive learning after re-scraping")
            adaptive_response = adaptive_learner.adaptive_response(
                query=query,
                max_length=1024,
                max_new_tokens=50
            )
            if verbose:
                print("\n--- Adaptive Response After Re-scraping ---")
                print(adaptive_response)

            logger.info("Fact-checking the response after re-scraping")
            fact_checked_data = fact_checker.check(query, adaptive_response)
            if isinstance(fact_checked_data, tuple):
                is_misinformation, fact_check_details = fact_checked_data
                if not is_misinformation:
                    new_answer = adaptive_response
                else:
                    new_answer = adaptive_response + "\n\nNote: This information may require further verification."
            else:
                new_answer = adaptive_response + "\n\nNote: Unable to verify the information."
            if verbose:
                print("\n--- Fact-checked Response After Re-scraping ---")
                print(new_answer)

            logger.info("Enhancing grammar after re-scraping")
            new_answer = grammar_enhancer.enhance_grammar(query, new_answer)

            new_answer = regex_cleaner.clean_text(new_answer.strip())

        if verbose:
            print(f"\n--- Query Result ---\n{new_answer}\n--- End of Result ---")

        cache.set(query, new_answer)

        print_cleaned_text(new_answer)

        logger.info("End of query processing")

        # Get user feedback
        user_feedback = input("Was this answer helpful? (yes/no): ").lower()

        # Store the answer as learned information only if the user found it helpful
        if user_feedback == 'yes':
            adaptive_learner.learned_information = [{"query": query, "response": new_answer}]
            grammar_enhancer.learn_from_feedback(query, adaptive_response, new_answer, user_feedback)

        # Output learned information
        print("\n--- Learned Information ---")
        learned_info = adaptive_learner.get_learned_information()
        if learned_info:
            for item in learned_info:
                print(f"Query: {item['query']}")
                print(f"Response: {item['response']}")
                print("-" * 50)
        else:
            print("No new information was learned from this interaction.")
        print("\n--- End of Learned Information ---")

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        print_cleaned_text("An unexpected error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    asyncio.run(main())