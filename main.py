#!/usr/bin/env python

import argparse
import logging
import asyncio
import time
import os
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
import nltk
from nltk.corpus import stopwords
from database_handler import DatabaseHandler
from web_scraper import WebScraper
from data_analyzer import DataAnalyzer
import json
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

warnings.filterwarnings("ignore", category=UserWarning, message="Your max_length is set to")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

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
    os.system('clear' if os.name == 'posix' else 'cls')  # Clear the terminal screen
    print(text)

def extract_text_content(item):
    if isinstance(item, dict) and 'content' in item:
        content = item['content']
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return ' '.join(str(c) for c in content if c)
    elif isinstance(item, list):
        return ' '.join(str(c) for c in item if c)
    elif isinstance(item, str):
        return item
    return ''

class AdvancedNLPProcessor:
    def __init__(self, model_name="google/flan-t5-large", sentence_model_name='all-MiniLM-L6-v2', summarizer_model_name="facebook/bart-large-cnn"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.sentence_model = SentenceTransformer(sentence_model_name).to(self.device)
        
        self.summarizer = pipeline("summarization", model=summarizer_model_name, device=0 if self.device.type == "cuda" else -1)
        
        self.nlp = spacy.load("en_core_web_sm")
        
        self.stop_words = set(stopwords.words('english'))

    def generate_answer(self, query: str, context: str) -> str:
        input_text = f"Answer the following question based on the given context. If the context doesn't contain enough information, say so and provide the best possible answer based on general knowledge.\nQuestion: {query}\nContext: {context}"
        inputs = self.qa_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.qa_model.generate(**inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        
        return self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def semantic_search(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        query_embedding = self.sentence_model.encode(query, convert_to_tensor=True)
        document_embeddings = self.sentence_model.encode(documents, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(query_embedding, document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))
        
        return [documents[idx] for idx in top_results.indices]

    def extract_key_information(self, text: str) -> str:
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Remove stopwords and lemmatize
        processed_sentences = [
            ' '.join([token.lemma_ for token in self.nlp(sentence) if token.text.lower() not in self.stop_words])
            for sentence in sentences
        ]
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords for each sentence
        keywords = []
        for i, sentence in enumerate(processed_sentences):
            tfidf_row = tfidf_matrix[i].toarray()[0]
            sorted_items = sorted(zip(tfidf_row, feature_names), key=lambda x: x[0], reverse=True)
            keywords.append([item[1] for item in sorted_items[:5]])
        
        # Score sentences based on keywords
        sentence_scores = [sum([1 for keyword in sentence_keywords if keyword in ' '.join(keywords).split()]) for sentence_keywords in keywords]
        
        # Select top sentences
        top_sentence_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:3]
        key_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
        
        return ' '.join(key_sentences) if key_sentences else "No key information extracted."

    def entity_recognition(self, text: str) -> List[Dict[str, str]]:
        doc = self.nlp(text)
        return [{'entity': ent.text, 'label': ent.label_} for ent in doc.ents]

    def summarize(self, text: str) -> str:
        return self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    def process(self, query: str, documents: List[str]) -> Dict:
        # Flatten and ensure all items are strings
        flat_documents = []
        for doc in documents:
            if isinstance(doc, list):
                flat_documents.extend([str(item) for item in doc if item])
            elif doc:
                flat_documents.append(str(doc))
        
        if not flat_documents:
            return {
                'query': query,
                'answer': "I'm sorry, but I couldn't find any relevant information to answer your question.",
                'key_information': "",
                'entities': [],
                'summary': ""
            }
        
        relevant_docs = self.semantic_search(query, flat_documents)
        context = " ".join(relevant_docs)
        
        answer = self.generate_answer(query, context)
        key_info = self.extract_key_information(answer)
        entities = self.entity_recognition(answer)
        summary = self.summarize(answer)
        
        return {
            'query': query,
            'answer': answer,
            'key_information': key_info,
            'entities': entities,
            'summary': summary
        }

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
    logging.info(f"Number of results to retrieve: {num_results}")

    device = get_device()

    db_handler = DatabaseHandler(db_name)
    web_scraper = WebScraper()
    data_analyzer = DataAnalyzer(db_name)
    nlp_processor = AdvancedNLPProcessor()

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

        # Extract text content from scraped data
        texts = []
        for item in scraped_data:
            extracted_text = extract_text_content(item)
            if extracted_text:
                texts.append(extracted_text)
        
        if not texts:
            print_cleaned_text("I'm sorry, but I couldn't extract any meaningful text from the search results.")
            return

        total_text_length = sum(len(str(t)) for t in texts)

        result = nlp_processor.process(query, texts)
        
        final_response = result['answer']
        final_response = remove_redundant_sentences(final_response)
        final_response = remove_exact_duplicates(final_response)
        print_cleaned_text(final_response)
        
        db_handler.store_data({'query': query, 'response': final_response})
        db_handler.save_metrics(query=query, num_results=num_results, total_text_length=total_text_length, 
                                total_summary_length=len(result['summary']), processing_time=processing_time, 
                                metric_name='', metric_value=0)
        
        logging.info(f"Summary Length: {len(result['summary'])}")
        logging.info(f"Text Length: {total_text_length}")
        logging.info(f"Processing Time: {processing_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        print_cleaned_text(f"An error occurred while processing your request: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
