#!/usr/bin/env python3

import logging
import os
import re
import json
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from web_scraper import WebScraper
from fact_checker import FactChecker
from database_handler import DatabaseHandler
import textwrap
from regex_cleaner import RegexCleaner

# Download necessary NLTK data
import nltk
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerEnhancer:
    def __init__(self, device=None):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        self.scraper = WebScraper(device=device)
        self.db_handler = DatabaseHandler()
        self.fact_checker = FactChecker(db_handler=self.db_handler, device=device)
        self.regex_cleaner = RegexCleaner(exclusion_files=self.db_handler.exclusion_files)
        self.training_data = self.load_all_training_data()

    def load_training_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                training_data = json.load(f)
            logger.info(f"Training data loaded from {file_path}")
            return training_data
        except Exception as e:
            logger.error(f"Error loading training data from {file_path}: {e}")
            return {}

    def load_all_training_data(self):
        all_training_data = {}
        training_dir = 'training_data'
        for file in os.listdir(training_dir):
            if file.endswith('.json'):
                file_path = os.path.join(training_dir, file)
                data = self.load_training_data(file_path)
                key = os.path.splitext(file)[0]
                all_training_data[key] = data
        return all_training_data

    def enhance_answer(self, query, original_answer):
        logger.info("Enhancing answer...")
        try:
            if not original_answer:
                logger.warning("Original answer is empty, returning as is.")
                return original_answer

            # Stage 1: Gross filtering
            cleaned_answer = self.regex_cleaner.clean_text(original_answer)
            if len(cleaned_answer) == 0:
                logger.error("Answer is empty after gross filtering.")
                return original_answer

            # Stage 2: Moderate filtering (if necessary)
            moderately_cleaned_answer = cleaned_answer  # No additional stage-specific filtering for now

            # Stage 3: Fine correction
            readable_answer = self.improve_readability(moderately_cleaned_answer)
            verified_answer = self.fact_check(query, readable_answer)

            if verified_answer.startswith("Warning:"):
                final_answer = self.ensure_relevance(query, verified_answer[verified_answer.index(':')+1:].strip())
                final_answer = "Warning: This information may not be accurate.\n\n" + final_answer
            else:
                final_answer = self.ensure_relevance(query, verified_answer)

            return self.format_long_answer(final_answer)
        except Exception as e:
            logger.error(f"Error in enhance_answer: {e}")
            return self.format_long_answer(original_answer)

    def ensure_relevance(self, query, answer):
        prompt = f"Question: {query}\nOriginal Answer: {answer}\nPlease provide a concise and accurate answer to the question, focusing only on relevant information."

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output = self.model.generate(
            input_ids,
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        relevant_answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        relevant_answer = ' '.join(relevant_answer.split())
        return relevant_answer

    def fact_check(self, query, answer):
        logger.info("Fact-checking the answer...")
        try:
            is_misinformation, fact_check_details = self.fact_checker.check(query, answer)
            if is_misinformation:
                logger.warning(f"Answer flagged as potential misinformation: {fact_check_details}")
                return f"Warning: This information may not be accurate. {answer}"
            else:
                logger.info("Answer passed fact-checking.")
                return answer
        except Exception as e:
            logger.error(f"Error in fact-checking: {e}")
            return answer

    def improve_readability(self, text):
        sentences = sent_tokenize(text)
        improved_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 20:
                parts = self.split_long_sentence(sentence)
                improved_sentences.extend(parts)
            else:
                improved_sentences.append(sentence)
        improved_text = ' '.join(improved_sentences)
        improved_text = self.correct_punctuation_spacing(improved_text)
        return improved_text

    def split_long_sentence(self, sentence):
        words = sentence.split()
        mid = len(words) // 2
        return [' '.join(words[:mid]), ' '.join(words[mid:])]

    def correct_punctuation_spacing(self, text):
        text = re.sub(r'\s+([?.!,"])', r'\1', text)
        text = re.sub(r'([?.!,"])', r'\1 ', text).strip()
        return text

    def format_long_answer(self, answer, max_width=80):
        formatted_lines = textwrap.wrap(answer, width=max_width)
        return '\n'.join(formatted_lines)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer Enhancement Script")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to get an answer for")
    parser.add_argument("--original_answer", "-a", type=str, required=True, help="The original answer to enhance")
    args = parser.parse_args()

    enhancer = AnswerEnhancer()
    enhanced_answer = enhancer.enhance_answer(args.query, args.original_answer)
    print(enhanced_answer)
