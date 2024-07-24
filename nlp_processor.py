#!/usr/bin/env python

import os
import logging
from functools import lru_cache
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NLPProcessor:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate_response(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        output = self.model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

@lru_cache(maxsize=2)
def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class RetrievalAugmentedGeneration:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
        self.texts = []

    def add_to_knowledge_base(self, texts):
        self.texts.extend(texts)
        if self.texts:
            self.vectorizer.fit(self.texts)

    def retrieve(self, query, k=5):
        if not self.texts:
            return []
        query_vector = self.vectorizer.transform([query])
        text_vectors = self.vectorizer.transform(self.texts)
        similarities = cosine_similarity(query_vector, text_vectors)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.texts[i] for i in top_k_indices]

rag = RetrievalAugmentedGeneration()

def process_nlp(data, query):
    logger.info("Starting NLP processing...")
    model_name = 'google/flan-t5-large'
    
    try:
        model, tokenizer = load_model(model_name)
        ner_pipeline = pipeline("ner")
        sentiment_pipeline = pipeline("sentiment-analysis")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return f"Failed to load model {model_name}. Error: {e}"
    
    try:
        # Add scraped data to knowledge base
        content_list = [item['content'] for item in data if isinstance(item.get('content'), str) and item['content'].strip()]
        logger.info(f"Content list: {content_list}")
        rag.add_to_knowledge_base(content_list)
        
        # Retrieve relevant information
        retrieved_info = rag.retrieve(query)
        logger.info(f"Retrieved info: {retrieved_info}")
        context = " ".join([str(info) for info in retrieved_info])  # Ensure all items are strings
        
        if not context:
            context = "No relevant information found."
        
        # Prepare input text
        input_text = f"Answer the following question based on the given context. If the context doesn't contain enough information, say so and provide the best possible answer based on general knowledge.\nQuestion: {query}\nContext: {context}"
        
        # Generate response
        inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
        outputs = model.generate(**inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Named Entity Recognition
        entities = ner_pipeline(response)
        
        # Sentiment Analysis
        sentiment = sentiment_pipeline(response)[0]
        
        # Summarization
        summary = summarizer(response, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        
        # Extract key information
        key_info = extract_key_information(response)
        
        # Prepare processed response
        processed_response = {
            'query': query,
            'response': response,
            'entities': entities,
            'sentiment': sentiment,
            'summary': summary,
            'key_info': key_info
        }
        
        logger.info("NLP processing completed.")
        return processed_response
    except Exception as e:
        logger.error(f"Error during NLP processing: {e}")
        logger.error(f"Data: {data}")
        logger.error(f"Query: {query}")
        return f"Error during NLP processing: {e}"

def extract_key_information(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return "No key information extracted."
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in sentences]
    
    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top keywords based on TF-IDF scores
    def get_top_keywords(tfidf_row, feature_names, top_n=5):
        sorted_items = sorted(zip(tfidf_row, feature_names), key=lambda x: x[0], reverse=True)
        return [item[1] for item in sorted_items[:top_n]]
    
    keywords = [get_top_keywords(tfidf_matrix[i].toarray()[0], feature_names) for i in range(len(sentences))]
    
    # Score sentences based on keywords
    sentence_scores = [sum([1 for keyword in sentence_keywords if keyword in ' '.join(keywords).split()]) for sentence_keywords in keywords]
    
    # Select top sentences
    top_sentence_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:3]
    key_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    
    return ' '.join(key_sentences) if key_sentences else "No key information extracted."

if __name__ == "__main__":
    # This main block is for testing purposes only
    test_data = [
        {'content': 'This is a sample content for testing purposes.'},
        {'content': 'It should be able to handle any type of query.'},
        {'content': 'The system is designed to be flexible and adaptable.'}
    ]
    test_query = "What can this system do?"
    result = process_nlp(test_data, test_query)
    print(result)