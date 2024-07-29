#!/usr/bin/env python3

import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from database_handler import DatabaseHandler
from fact_checker import FactChecker
from web_scraper import WebScraper
import textwrap

logger = logging.getLogger(__name__)

class AdaptiveLearning:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.db_handler = DatabaseHandler()
        self.web_scraper = WebScraper()
        self.fact_checker = FactChecker(self.db_handler)
        self.load_model(model_name)
        self.learned_information = []
        logger.info(f"AdaptiveLearning initialized using device: {self.device}")

    def load_model(self, model_name):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Model {model_name} loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def update_model(self, query, response, feedback_score):
        logger.info(f"Received feedback: {feedback_score} for query: {query} and response: {response}")
        inputs = self.tokenizer(query + self.tokenizer.sep_token + response, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        labels = torch.tensor([feedback_score]).unsqueeze(0).to(self.device)
        
        self.model.train()
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        optimizer.step()
        optimizer.zero_grad()

    def generate_answer(self, sources, max_length=1024, max_new_tokens=512):
        combined_content = " ".join([source.get('content', '') for source in sources if source.get('content')])
        if not combined_content:
            logger.warning("No valid content found in sources")
            return "No valid content found to generate an answer."
        
        try:
            inputs = self.tokenizer.encode(combined_content, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            formatted_answer = self.format_answer(answer)
            logger.info(f"Generated answer: {formatted_answer}")
            return formatted_answer
        except IndexError:
            logger.error("IndexError in generate_answer. Falling back to simple summarization.")
            return self.simple_summarize(combined_content)
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            return f"Error generating answer: {str(e)}"

    def simple_summarize(self, text, max_length=100):
        words = text.split()
        if len(words) <= max_length:
            return text
        return ' '.join(words[:max_length]) + "..."

    def format_answer(self, answer, max_width=80):
        formatted_lines = textwrap.wrap(answer, width=max_width)
        return '\n'.join(formatted_lines)

    def fetch_sources(self, query, num_sources=5):
        logger.info(f"Fetching sources for query: {query}")
        sources, _ = self.web_scraper.scrape(query, num_sources)
        logger.info(f"Fetched {len(sources)} sources")
        return sources

    def extract_unique_sources(self, first_sources, new_sources):
        first_contents = {source.get('content', '') for source in first_sources}
        unique_sources = [source for source in new_sources if source.get('content', '') not in first_contents]
        logger.info(f"Extracted {len(unique_sources)} unique new sources")
        return unique_sources

    def adaptive_response(self, query, max_length=1024, max_new_tokens=512):
        logger.info(f"Starting adaptive response for query: {query}")
        first_sources = self.fetch_sources(query)
        initial_answer = self.generate_answer(first_sources, max_length=max_length, max_new_tokens=max_new_tokens)
        
        if initial_answer.startswith("Error generating answer") or initial_answer == "No valid content found to generate an answer.":
            return initial_answer

        is_misinformation, fact_check_details = self.fact_checker.check(initial_answer, query)
        if not is_misinformation:
            self.learned_information.append({"query": query, "response": initial_answer})
            return initial_answer

        logger.warning(f"Initial answer flagged as misinformation: {fact_check_details}")

        attempt = 0
        max_attempts = 5
        double_checked_answer = initial_answer

        while attempt < max_attempts:
            attempt += 1
            logger.info(f"Attempt {attempt} to fetch a new answer")

            new_sources = self.fetch_sources(query)
            unique_new_sources = self.extract_unique_sources(first_sources, new_sources)
            if not unique_new_sources:
                logger.error("Unable to fetch unique sources different from the initial ones.")
                return "Unable to fetch unique sources different from the initial ones."

            double_checked_answer = self.generate_answer(unique_new_sources, max_length=max_length, max_new_tokens=max_new_tokens)
            
            if double_checked_answer.startswith("Error generating answer") or double_checked_answer == "No valid content found to generate an answer.":
                continue

            is_misinformation, fact_check_details = self.fact_checker.check(double_checked_answer, query)
            if not is_misinformation:
                self.learned_information.append({"query": query, "response": double_checked_answer})
                break

        if is_misinformation:
            logger.error("Double-checked answer also flagged as misinformation.")
            return f"Warning: The answer is considered too unreliable.\nAnswer: {double_checked_answer}"

        return double_checked_answer

    def get_learned_information(self):
        return self.learned_information

    def load_latest_model(self):
        try:
            self.load_model("./adaptive_model")
            logger.info("Loaded the latest adaptive model")
        except Exception as e:
            logger.warning(f"Failed to load the latest adaptive model: {e}")
            logger.info("Using the initial model.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Learning Script")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to get an answer for")
    args = parser.parse_args()

    adaptive_learner = AdaptiveLearning()
    query = args.query
    improved_response = adaptive_learner.adaptive_response(query)
    print(f"Improved response:\n{improved_response}")
    
    print("\nLearned Information:")
    for item in adaptive_learner.get_learned_information():
        print(f"Query: {item['query']}")
        print(f"Response: {item['response']}")
        print("-" * 50)