#!/usr/bin/env python3

import faiss
import logging
import numpy as np
import re
import time
import torch
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import SIMILARITY_THRESHOLD, CACHE_SIZE, BATCH_SIZE, MAX_WORKERS, INDEX_UPDATE_FREQUENCY, USE_FAISS
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Dict, Tuple, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class SimilarityChecker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        self.embedding_cache = LRUCache(CACHE_SIZE)
        self.index = None
        self.use_faiss = USE_FAISS
        if self.use_faiss:
            self.initialize_faiss_index()
        self.normalize_pattern = re.compile(r'[^a-z\s]')
        self.stop_words = re.compile(r'\b(tell me|can you|please)\b')

    def initialize_faiss_index(self):
        config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        dimension = config.hidden_size
        self.index = faiss.IndexFlatIP(dimension)
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def normalize_query(self, text: str) -> str:
        text = text.lower()
        text = self.normalize_pattern.sub('', text)
        text = self.stop_words.sub('', text)
        return text.strip()

    @torch.no_grad()
    def get_embedding(self, text: str) -> torch.Tensor:
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding.to(self.device)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
        self.embedding_cache.put(text, embedding)
        return embedding.to(self.device)

    @torch.no_grad()
    def batch_process_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding.to(self.device))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            batch_embeddings = self.sentence_transformer.encode(uncached_texts, convert_to_tensor=True, show_progress_bar=False).to(self.device)
            for i, embedding in zip(uncached_indices, batch_embeddings):
                self.embedding_cache.put(texts[i], embedding.cpu())
                embeddings.insert(i, embedding)

        if embeddings:
            embeddings = [e.squeeze() if e.dim() > 1 else e for e in embeddings]
            return torch.stack(embeddings).to(self.device)
        else:
            return torch.tensor([], device=self.device)

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b)

    def find_similar_interaction(self, query: str, interactions: List[Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], float]:
        normalized_query = self.normalize_query(query)
        query_embedding = self.get_embedding(normalized_query)
        
        interaction_queries = [self.normalize_query(interaction["query"]) for interaction in interactions]
        interaction_embeddings = self.batch_process_embeddings(interaction_queries)
        
        if interaction_embeddings.size(0) > 0:
            similarities = self.cosine_similarity(query_embedding, interaction_embeddings)
            max_similarity, max_index = torch.max(similarities, dim=0)
            
            if max_similarity > SIMILARITY_THRESHOLD:
                similar_interaction = interactions[max_index]
                if similar_interaction["answer"] == "I don't have a cached answer for this query.":
                    for i, sim in enumerate(similarities):
                        if i != max_index and sim > SIMILARITY_THRESHOLD and interactions[i]["answer"] != "I don't have a cached answer for this query.":
                            return interactions[i], sim.item()
                return similar_interaction, max_similarity.item()
        
        return None, 0.0

    def find_top_k_similar(self, query: str, interactions: List[Dict[str, str]], k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        normalized_query = self.normalize_query(query)
        query_embedding = self.get_embedding(normalized_query)
        
        interaction_queries = [self.normalize_query(interaction["query"]) for interaction in interactions]
        interaction_embeddings = self.batch_process_embeddings(interaction_queries)
        
        if self.use_faiss:
            D, I = self.index.search(query_embedding.unsqueeze(0).cpu().numpy(), k)
            return [(interactions[i], D[0][j]) for j, i in enumerate(I[0])]
        else:
            similarities = self.cosine_similarity(query_embedding, interaction_embeddings)
            top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(interactions)))
            return [(interactions[i], v.item()) for i, v in zip(top_k_indices, top_k_values)]

    def update_embeddings(self, new_interactions: List[Dict[str, str]]):
        for interaction in new_interactions:
            normalized_query = self.normalize_query(interaction["query"])
            if self.embedding_cache.get(normalized_query) is None:
                _ = self.get_embedding(normalized_query)
        
        if self.use_faiss and len(new_interactions) >= INDEX_UPDATE_FREQUENCY:
            self.update_faiss_index(new_interactions)

    def update_faiss_index(self, new_interactions: List[Dict[str, str]]):
        new_embeddings = self.batch_process_embeddings([self.normalize_query(interaction["query"]) for interaction in new_interactions])
        self.index.add(new_embeddings.cpu().numpy())

    def clear_embedding_cache(self):
        self.embedding_cache = LRUCache(CACHE_SIZE)
        torch.cuda.empty_cache()

    def compute_similarity_matrix(self, queries: List[str]) -> np.ndarray:
        normalized_queries = [self.normalize_query(query) for query in queries]
        embeddings = self.batch_process_embeddings(normalized_queries)
        return sklearn_cosine_similarity(embeddings.cpu().numpy())

    @lru_cache(maxsize=1000)
    def cached_similarity(self, query1: str, query2: str) -> float:
        normalized_query1 = self.normalize_query(query1)
        normalized_query2 = self.normalize_query(query2)
        embedding1 = self.get_embedding(normalized_query1)
        embedding2 = self.get_embedding(normalized_query2)
        return self.cosine_similarity(embedding1, embedding2).item()

    def process_queries_in_parallel(self, queries: List[str], interactions: List[Dict[str, str]], func: str, **kwargs) -> List:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            if func == "find_similar_interaction":
                futures = [executor.submit(self.find_similar_interaction, query, interactions) for query in queries]
            elif func == "find_top_k_similar":
                k = kwargs.get('k', 5)
                futures = [executor.submit(self.find_top_k_similar, query, interactions, k) for query in queries]
            else:
                raise ValueError(f"Unknown function: {func}")
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def benchmark(self, queries: List[str], interactions: List[Dict[str, str]], runs: int = 5):
        logger.info(f"Running benchmark with {len(queries)} queries and {len(interactions)} interactions")
        
        for func in ["find_similar_interaction", "find_top_k_similar"]:
            total_time = 0
            for _ in range(runs):
                start_time = time.time()
                self.process_queries_in_parallel(queries, interactions, func)
                end_time = time.time()
                total_time += end_time - start_time
            
            avg_time = total_time / runs
            logger.info(f"Average time for {func}: {avg_time:.4f} seconds")

similarity_checker = SimilarityChecker()

def find_similar_interaction(query: str, interactions: List[Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], float]:
    return similarity_checker.find_similar_interaction(query, interactions)

def find_top_k_similar(query: str, interactions: List[Dict[str, str]], k: int = 5) -> List[Tuple[Dict[str, str], float]]:
    return similarity_checker.find_top_k_similar(query, interactions, k)

def update_embeddings(new_interactions: List[Dict[str, str]]):
    similarity_checker.update_embeddings(new_interactions)

def clear_embedding_cache():
    similarity_checker.clear_embedding_cache()

def compute_similarity_matrix(queries: List[str]) -> np.ndarray:
    return similarity_checker.compute_similarity_matrix(queries)

def cached_similarity(query1: str, query2: str) -> float:
    return similarity_checker.cached_similarity(query1, query2)

def process_queries_in_parallel(queries: List[str], interactions: List[Dict[str, str]], func: str, **kwargs) -> List:
    return similarity_checker.process_queries_in_parallel(queries, interactions, func, **kwargs)

def benchmark(queries: List[str], interactions: List[Dict[str, str]], runs: int = 5):
    similarity_checker.benchmark(queries, interactions, runs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Similarity Checker CLI")
    parser.add_argument("--query", type=str, help="Query to find similar interactions for")
    parser.add_argument("--interactions_file", type=str, help="JSON file containing interactions")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top similar interactions to return")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--benchmark_queries_file", type=str, help="JSON file containing benchmark queries")
    parser.add_argument("--benchmark_runs", type=int, default=5, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    import json
    
    if args.interactions_file:
        with open(args.interactions_file, 'r') as f:
            interactions = json.load(f)
    else:
        interactions = []
    
    if args.query:
        similar_interaction, similarity_score = find_similar_interaction(args.query, interactions)
        if similar_interaction:
            print(f"Similar interaction found:")
            print(f"Query: {similar_interaction['query']}")
            print(f"Answer: {similar_interaction['answer']}")
            print(f"Similarity score: {similarity_score}")
        else:
            print("No similar interaction found.")
        
        top_k_similar = find_top_k_similar(args.query, interactions, args.top_k)
        print(f"\nTop {args.top_k} similar interactions:")
        for interaction, score in top_k_similar:
            print(f"Query: {interaction['query']}")
            print(f"Answer: {interaction['answer']}")
            print(f"Similarity score: {score}")
            print()
    
    if args.benchmark:
        if args.benchmark_queries_file:
            with open(args.benchmark_queries_file, 'r') as f:
                benchmark_queries = json.load(f)
        else:
            benchmark_queries = [interaction['query'] for interaction in interactions[:10]]
        
        benchmark(benchmark_queries, interactions, args.benchmark_runs)