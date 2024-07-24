#!/usr/bin/env python

import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from typing import List, Dict
import numpy as np
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class AdvancedNLPProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(self.device)
        self.qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if self.device.type == "cuda" else -1)
        
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
        relevant_docs = self.semantic_search(query, documents)
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

if __name__ == "__main__":
    processor = AdvancedNLPProcessor()
    
    # Test the processor
    query = input("Enter your question: ")
    documents = [
        "This is a sample document to test the advanced NLP processor.",
        "It should be able to handle any type of question.",
        "The processor uses advanced techniques like semantic search and named entity recognition.",
        "It also provides summaries and extracts key information from the generated answer.",
        "This should result in more comprehensive and higher-quality responses to user queries."
    ]
    
    result = processor.process(query, documents)
    print(json.dumps(result, indent=2))
