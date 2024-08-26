#!/usr/bin/env python3

import argparse
import concurrent.futures
import nltk
import numpy as np
import re
import sys
import os
import torch
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure the Hugging Face cache directory is set
CACHE_DIR = os.path.expanduser('/home/jman/.cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

# Check and set CUDA device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize SentenceTransformer model with cache
model_name = 'all-MiniLM-L6-v2'
model_path = os.path.join(CACHE_DIR, model_name)

if not os.path.exists(model_path):
    logger.info(f"Downloading the model {model_name} to {model_path}")
    model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
else:
    logger.info(f"Loading the model {model_name} from {model_path}")
    model = SentenceTransformer(model_path)

model.to(device)
logger.info(f"Loaded pretrained SentenceTransformer: {model_name} from {model_path}")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Precompile regex pattern
non_alpha_pattern = re.compile(r'[^a-zA-Z\s]')

@lru_cache(maxsize=1000)
def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing special characters,
    converting to lowercase, tokenizing, removing stopwords, and lemmatizing.
    """
    text = non_alpha_pattern.sub('', text.lower())
    tokens = word_tokenize(text)
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(preprocessed_tokens)

def compute_similarity(query: str, answer: str, tfidf_weight: float = 0.5, st_weight: float = 0.5) -> float:
    """
    Compute the similarity between the query and answer using a combination
    of TF-IDF cosine similarity and SentenceTransformer embeddings.
    """
    if not (0 <= tfidf_weight <= 1) or not (0 <= st_weight <= 1) or tfidf_weight + st_weight != 1.0:
        raise ValueError("tfidf_weight and st_weight must be between 0 and 1, and their sum must be 1.")

    preprocessed_query = preprocess_text(query)
    preprocessed_answer = preprocess_text(answer)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_query, preprocessed_answer])
    tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Use the model's built-in batch encoding to speed up computation
    embeddings = model.encode([query, answer], show_progress_bar=False)
    st_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return tfidf_weight * tfidf_similarity + st_weight * st_similarity

def find_most_similar(query: str, corpus: List[str], top_k: int = 5, tfidf_weight: float = 0.5, st_weight: float = 0.5) -> List[Tuple[int, float]]:
    """
    Find the most similar sentences in the corpus to the given query.
    Returns a list of tuples containing the index and similarity score.
    """
    if not corpus:
        raise ValueError("The corpus cannot be empty.")
    
    query_embedding = model.encode([query], show_progress_bar=False)
    corpus_embeddings = model.encode(corpus, show_progress_bar=False)
    
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(idx, similarities[idx]) for idx in top_indices]

def batch_compute_similarity(queries: List[str], answers: List[str], tfidf_weight: float = 0.5, st_weight: float = 0.5) -> List[float]:
    """
    Compute similarities for multiple query-answer pairs in batch.
    """
    if len(queries) != len(answers):
        raise ValueError("The number of queries and answers must be the same.")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        preprocessed_queries, preprocessed_answers = zip(
            *executor.map(lambda pair: (preprocess_text(pair[0]), preprocess_text(pair[1])), zip(queries, answers))
        )
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(list(preprocessed_queries) + list(preprocessed_answers))
    tfidf_similarities = cosine_similarity(tfidf_matrix[:len(queries)], tfidf_matrix[len(queries):]).diagonal()
    
    query_embeddings = model.encode(queries, show_progress_bar=False)
    answer_embeddings = model.encode(answers, show_progress_bar=False)
    st_similarities = cosine_similarity(query_embeddings, answer_embeddings).diagonal()
    
    return [tfidf_weight * tfidf_sim + st_weight * st_sim for tfidf_sim, st_sim in zip(tfidf_similarities, st_similarities)]

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Compute similarity between queries and answers.")
    parser.add_argument("query", type=str, help="The query text.")
    parser.add_argument("answer", type=str, help="The answer text.")
    parser.add_argument("--corpus", nargs="*", help="Optional corpus sentences to find the most similar ones.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top similar sentences to retrieve.")
    parser.add_argument("--tfidf_weight", type=float, default=0.5, help="Weight for TF-IDF similarity.")
    parser.add_argument("--st_weight", type=float, default=0.5, help="Weight for SentenceTransformer similarity.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    try:
        similarity = compute_similarity(args.query, args.answer, args.tfidf_weight, args.st_weight)
        print(f"Similarity between query and answer: {similarity:.4f}")
        
        if args.corpus:
            similar_sentences = find_most_similar(args.query, args.corpus, args.top_k, args.tfidf_weight, args.st_weight)
            print("\nMost similar sentences in corpus:")
            for idx, score in similar_sentences:
                print(f"- {args.corpus[idx]} (Similarity: {score:.4f})")
        else:
            print("No corpus provided for similarity comparison.")

        # Batch similarity example
        queries = [args.query]  # Add more queries if needed
        answers = [args.answer]  # Add more answers if needed
        batch_similarities = batch_compute_similarity(queries, answers, args.tfidf_weight, args.st_weight)
        print("\nBatch similarities:")
        for q, a, s in zip(queries, answers, batch_similarities):
            print(f"Query: {q}")
            print(f"Answer: {a}")
            print(f"Similarity: {s:.4f}\n")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)