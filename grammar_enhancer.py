#!/usr/bin/env python3

import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from database_handler import DatabaseHandler
from nltk.tokenize import sent_tokenize
import nltk
from diskcache import Cache

nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)
cache = Cache('model_cache')

class GrammarEnhancer:
    def __init__(self, db_name='ai_assistant.db'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_handler = DatabaseHandler(db_name)
        self.models = self.load_models()
        self.irrelevant_phrases = self.load_irrelevant_phrases()

    def load_models(self):
        models = {}
        model_names = [
            "facebook/bart-large-cnn",
            "t5-base",
        ]
        for name in model_names:
            model, tokenizer = cache.get(name, (None, None))
            if model is None or tokenizer is None:
                model = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir='model_cache').to(self.device)
                tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='model_cache')
                cache.set(name, (model, tokenizer))
            models[name] = {'model': model, 'tokenizer': tokenizer}
        return models

    def load_irrelevant_phrases(self):
        # Load irrelevant phrases from the database
        return self.db_handler.get_irrelevant_phrases()

    def enhance_grammar(self, query, answer):
        enhanced_answers = []
        for name, model_data in self.models.items():
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            
            input_text = f"Improve grammar and relevance: Query: {query} Answer: {answer}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            output = model.generate(input_ids, max_length=512, num_return_sequences=1, no_repeat_ngram_size=2)
            enhanced = tokenizer.decode(output[0], skip_special_tokens=True)
            enhanced_answers.append(enhanced)
        
        # Combine and clean the enhanced answers
        combined_answer = self.combine_and_clean(enhanced_answers)
        
        # Ensure the answer is relevant to the query
        final_answer = self.ensure_relevance(query, combined_answer)
        
        # Store the enhanced answer for learning
        self.db_handler.store_enhanced_answer(query, answer, final_answer)
        
        return final_answer

    def combine_and_clean(self, answers):
        # Combine answers and remove duplicates
        all_sentences = []
        for answer in answers:
            all_sentences.extend(sent_tokenize(answer))
        unique_sentences = list(dict.fromkeys(all_sentences))
        
        # Remove irrelevant sentences
        relevant_sentences = [
            sent for sent in unique_sentences
            if not self.is_irrelevant(sent)
        ]
        
        return ' '.join(relevant_sentences)

    def is_irrelevant(self, sentence):
        return any(phrase in sentence.lower() for phrase in self.irrelevant_phrases)

    def ensure_relevance(self, query, answer):
        # Check if the answer is relevant to the query
        if not self.is_relevant(query, answer):
            relevant_info = self.db_handler.get_relevant_info(query)
            if relevant_info:
                answer = f"{answer}\n\nAdditional relevant information: {relevant_info}"
        return answer

    def is_relevant(self, query, answer):
        # Simple relevance check (can be improved with more sophisticated methods)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        return len(query_words.intersection(answer_words)) > 0

    def learn_from_feedback(self, query, original_answer, enhanced_answer, user_feedback):
        # Store user feedback for future improvements
        self.db_handler.store_feedback(query, original_answer, enhanced_answer, user_feedback)
        
        # Update irrelevant phrases based on feedback
        if user_feedback.lower() == 'negative':
            new_phrases = self.extract_potential_irrelevant_phrases(enhanced_answer)
            self.irrelevant_phrases.extend(new_phrases)
            self.db_handler.update_irrelevant_phrases(self.irrelevant_phrases)

    def extract_potential_irrelevant_phrases(self, text):
        # Extract potential irrelevant phrases (this is a simple implementation and can be improved)
        sentences = sent_tokenize(text)
        return [sent.lower() for sent in sentences if len(sent.split()) < 5]

if __name__ == "__main__":
    enhancer = GrammarEnhancer()
    # The main execution is now handled by main.py