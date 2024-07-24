#!/usr/bin/env python3

import logging
from nltk.tokenize import sent_tokenize
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class AnswerEnhancer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

    def enhance_answer(self, query, original_answer):
        logger.info("Enhancing answer...")
        try:
            if not original_answer:
                logger.warning("Original answer is empty, returning as is.")
                return original_answer

            # Improve readability
            readable_answer = self.improve_readability(original_answer)
            
            # Fact-check and verify
            verified_answer = self.fact_check(readable_answer)
            
            # Ensure the answer is relevant to the query
            final_answer = self.ensure_relevance(query, verified_answer)
            
            return final_answer
        except Exception as e:
            logger.error(f"Error in enhance_answer: {e}")
            return original_answer  # Return the original answer if enhancement fails

    def ensure_relevance(self, query, answer):
        prompt = f"Question: {query}\nOriginal Answer: {answer}\nPlease provide a concise and accurate answer to the question, focusing only on relevant information:"
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = self.model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        relevant_answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return relevant_answer

    def fact_check(self, answer):
        # This is a placeholder for a more sophisticated fact-checking mechanism
        # In a real-world scenario, you'd want to use a reliable fact-checking API or database
        logger.info("Fact-checking the answer")
        return answer  # For now, we're returning the original answer

    def improve_readability(self, text):
        sentences = sent_tokenize(text)
        improved_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 20:
                parts = self.split_long_sentence(sentence)
                improved_sentences.extend(parts)
            else:
                improved_sentences.append(sentence)
        return ' '.join(improved_sentences)

    def split_long_sentence(self, sentence):
        words = sentence.split()
        mid = len(words) // 2
        return [' '.join(words[:mid]), ' '.join(words[mid:])]

if __name__ == "__main__":
    # This main block is for testing purposes only
    enhancer = AnswerEnhancer()
    query = "What is the capital of France?"
    original_answer = "The capital of France is Paris, which is also the largest city in the country and serves as a major European cultural and economic center."
    enhanced_answer = enhancer.enhance_answer(query, original_answer)
    print(f"Original Answer: {original_answer}")
    print(f"\nEnhanced Answer: {enhanced_answer}")