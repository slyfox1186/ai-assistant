#!/usr/bin/env python3

import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import numpy as np
import spacy

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load Spacy model for named entity recognition
nlp = spacy.load('en_core_web_sm')

class ContextualEnhancer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = 'deepset/roberta-base-squad2'
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knowledge_graph = nx.Graph()

    def build_knowledge_graph(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            self.knowledge_graph.add_node(ent.text, label=ent.label_)
        for chunk in doc.noun_chunks:
            for token in chunk:
                if token.dep_ in ("amod", "compound"):
                    self.knowledge_graph.add_edge(chunk.root.text, token.text)
        return self.knowledge_graph

    def enhance_context(self, text, query):
        logger.info("Enhancing context using multiple techniques...")
        sentences = sent_tokenize(text)
        if not sentences:
            return text

        # Use TF-IDF for initial retrieval
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranked_sentences = [sentences[i] for i in similarities.argsort()[-5:][::-1]]

        # Use sentence transformers for semantic similarity
        sentence_embeddings = self.sentence_model.encode(sentences)
        query_embedding = self.sentence_model.encode([query])
        cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

        if cosine_scores.size(0) > 0:
            ranked_sentences_semantic = [sentences[i] for i in cosine_scores.argsort()[-5:].tolist()[::-1]]
        else:
            ranked_sentences_semantic = []

        # Combine results and build enhanced context
        combined_sentences = list(set(ranked_sentences + ranked_sentences_semantic))
        enhanced_context = ' '.join(combined_sentences)
        return enhanced_context

    def answer_question(self, context, question):
        inputs = self.tokenizer(question, context, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
        return answer

    def process(self, context, query):
        enhanced_context = self.enhance_context(context, query)
        answer = self.answer_question(enhanced_context, query)
        return answer

if __name__ == "__main__":
    enhancer = ContextualEnhancer()
    context = "Your context here..."
    query = "Your question here..."
    result = enhancer.process(context, query)
    print(f"Answer: {result}")
