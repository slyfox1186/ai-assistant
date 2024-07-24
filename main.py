#!/usr/bin/env python

import argparse
import logging
import asyncio
import time
import os
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
import torch
from database_handler import DatabaseHandler
from web_scraper import WebScraper
from data_analyzer import DataAnalyzer
from nlp_processor import NLPProcessor
import json
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore", category=UserWarning, message="Your max_length is set to")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    # Default to CPU since CUDA is not available
    device = torch.device("cpu")
    logging.info(f"Using CPU")
    return device

def split_text(text, max_length, tokenizer):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(tokenizer.encode(word, add_special_tokens=False))
        if current_length + word_length <= max_length:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def truncate_text(text, max_length, tokenizer):
    encoded_text = tokenizer.encode(text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(encoded_text[0], skip_special_tokens=True)

def summarize_text(batch, summarizer, tokenizer, max_length=1024):
    summaries = []
    for text in batch['text']:
        if isinstance(text, str):
            chunks = split_text(text, max_length, tokenizer)
            chunk_summaries = []
            for chunk in chunks:
                truncated_text = truncate_text(chunk, max_length, tokenizer)
                summary = summarizer(truncated_text, max_length=256, min_length=30, do_sample=False)[0]['summary_text']
                chunk_summaries.append(summary)
            summary = " ".join(chunk_summaries)
            summaries.append(summary)
        else:
            logging.warning(f"Unexpected data type in summarize_text: {type(text)}")
            summaries.append('')
    return {'summary': summaries}

def condense_summary(summary):
    sentences = summary.split('. ')
    condensed_summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else summary
    return condensed_summary

def remove_redundant_sentences(text):
    sentences = text.split('. ')
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)

    unique_sentences = []
    seen_sentences = set()

    for i, sentence in enumerate(sentences):
        if sentence in seen_sentences:
            continue
        unique_sentences.append(sentence)
        for j in range(i + 1, len(sentences)):
            if csim[i, j] > 0.8:  # Adjust threshold as needed
                seen_sentences.add(sentences[j])

    return '. '.join(unique_sentences)

def remove_exact_duplicates(text):
    sentences = text.split('. ')
    unique_sentences = []
    seen = set()

    for sentence in sentences:
        cleaned_sentence = sentence.strip().lower()
        if cleaned_sentence not in seen:
            seen.add(cleaned_sentence)
            unique_sentences.append(sentence)

    return '. '.join(unique_sentences)

def print_cleaned_text(text):
    os.system('clear')  # Clear the terminal screen
    print(text)

async def main():
    parser = argparse.ArgumentParser(description='Ask a question and get an answer by searching the web.')
    parser.add_argument('question', type=str, help='The question you want to ask.')
    parser.add_argument('--results', type=int, default=5, help='Number of search results to retrieve.')
    parser.add_argument('--db-name', type=str, default='ai_assistant.db', help='Database name')
    args = parser.parse_args()

    query = args.question
    num_results = args.results
    db_name = args.db_name

    logging.info(f"Searching for: {query}")

    # No CUDA device check, always use CPU
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # Use CPU

    db_handler = DatabaseHandler(db_name)
    web_scraper = WebScraper()
    data_analyzer = DataAnalyzer(db_name)
    nlp_processor = NLPProcessor()

    cached_answer = db_handler.retrieve_data(query)
    if cached_answer:
        print_cleaned_text(cached_answer)
        return

    try:
        start_time = time.time()
        scraped_data = web_scraper.scrape(query, num_results)
        processing_time = time.time() - start_time

        if not scraped_data:
            print_cleaned_text("I'm sorry, but I couldn't find any relevant information for your question.")
            return

        logging.info(f"Scraped data: {json.dumps(scraped_data, indent=4)}")

        labels = data_analyzer.analyze_data(scraped_data)
        logging.info(f"Data analysis complete. Labels assigned.")

        summarized_data = data_analyzer.summarize_content(scraped_data, query)
        
        texts = [item['content'] for item in scraped_data if isinstance(item.get('content'), str)]
        total_text_length = sum(len(t) for t in texts if isinstance(t, str))

        dataset = Dataset.from_dict({'text': texts})
        summarized_dataset = dataset.map(lambda batch: summarize_text(batch, summarizer, tokenizer), batched=True, batch_size=8)

        valid_summaries = [s for s in summarized_dataset['summary'] if s]

        if valid_summaries:
            combined_summary = ' '.join(valid_summaries)
            condensed_summary = condense_summary(combined_summary)
            cleaned_summary = remove_redundant_sentences(condensed_summary)
            cleaned_summary = remove_exact_duplicates(cleaned_summary)  # Ensuring exact duplicates are removed
            final_response = nlp_processor.generate_response(cleaned_summary)
            final_response = remove_exact_duplicates(final_response)
            print_cleaned_text(final_response)
            
            total_summary_length = sum(len(s) for s in valid_summaries)
            db_handler.store_data({'query': query, 'response': final_response})
            db_handler.save_metrics(query=query, num_results=num_results, total_text_length=total_text_length, 
                                    total_summary_length=total_summary_length, processing_time=processing_time, 
                                    metric_name='', metric_value=0)
            
            logging.info(f"Summary Length: {total_summary_length}")
            logging.info(f"Text Length: {total_text_length}")
            logging.info(f"Processing Time: {processing_time:.2f} seconds")
        else:
            print_cleaned_text("I'm sorry, but I couldn't extract any meaningful information from the search results. This could be due to network issues or the complexity of the question. Please try again or rephrase your question.")
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        print_cleaned_text(f"An error occurred while processing your request: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
