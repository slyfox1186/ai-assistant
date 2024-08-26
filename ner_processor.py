#!/usr/bin/env python3

import os
import torch
import spacy
import numpy as np
import nltk
import logging
import asyncio
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from flair.models import SequenceTagger
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from config import (NUM_BEAMS, NER_MAX_LENGTH, NER_CONFIDENCE_THRESHOLD, 
                    BATCH_SIZE, HF_CACHE_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define model names
ner_model_name = "jean-baptiste/roberta-large-ner-english"
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
flair_model_name = "flair/ner-english-ontonotes-large"
spacy_model_name = "en_core_web_trf"

class NERDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=NER_MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}

class NERProcessor:
    def __init__(self, ner_model_name: str, sentiment_model_name: str, flair_model_name: str, spacy_model_name: str, num_beams: int):
        self.ner_model_name = ner_model_name
        self.sentiment_model_name = sentiment_model_name
        self.flair_model_name = flair_model_name
        self.spacy_model_name = spacy_model_name
        self.num_beams = num_beams

        self.ner_tokenizer = None
        self.ner_model = None
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.nlp = None
        self.flair_ner = None

        # Initialize caches
        self.ner_cache = {}
        self.relation_cache = {}
        self.sentiment_cache = {}
        self.entity_linking_cache = {}

        self.lemmatizer = nltk.WordNetLemmatizer()

    def load_ner_model(self):
        if self.ner_model is None:
            logger.info(f"Loading NER model: {self.ner_model_name}")
            self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name, cache_dir=HF_CACHE_DIR)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_model_name, cache_dir=HF_CACHE_DIR).to(device)

    def load_sentiment_model(self):
        if self.sentiment_model is None:
            logger.info(f"Loading Sentiment model: {self.sentiment_model_name}")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name, cache_dir=HF_CACHE_DIR)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name, cache_dir=HF_CACHE_DIR).to(device)

    def load_spacy_model(self):
        if self.nlp is None:
            logger.info(f"Loading SpaCy model: {self.spacy_model_name}")
            self.nlp = spacy.load(self.spacy_model_name)

    def load_flair_model(self):
        if self.flair_ner is None:
            logger.info(f"Loading Flair model: {self.flair_model_name}")
            self.flair_ner = SequenceTagger.load(self.flair_model_name)

    def process_ner_batch(self, batch: Dict[str, torch.Tensor], ner_confidence_threshold: float) -> List[List[Dict[str, str]]]:
        self.load_ner_model()
        outputs = self.ner_model(**{k: v.to(device) for k, v in batch.items()})
        all_entities = []

        for i, logits in enumerate(outputs.logits):
            score = torch.nn.functional.softmax(logits, dim=-1)
            labels = torch.argmax(score, dim=-1)
            entities = []

            for j, (label, conf) in enumerate(zip(labels, score)):
                if label != 0 and conf[label] > ner_confidence_threshold:
                    try:
                        entity = {
                            'entity': self.ner_model.config.id2label[label.item()],
                            'score': conf[label].item(),
                            'index': j,
                            'word': self.ner_tokenizer.convert_ids_to_tokens(batch['input_ids'][i][j].item())
                        }
                        entities.append(entity)
                    except Exception as e:
                        logger.error(f"Error processing entity at index {j}: {str(e)}")

            all_entities.append(self.merge_entities(entities))

        return all_entities

    def merge_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not entities:
            return []

        merged = []
        current_entity = entities[0]

        for entity in entities[1:]:
            if (entity['entity'] == current_entity['entity'] and 
                entity['index'] == current_entity['index'] + 1):
                current_entity['index'] = entity['index']
                current_entity['word'] += entity['word'].lstrip('##')
                current_entity['score'] = (current_entity['score'] + entity['score']) / 2
            else:
                merged.append(current_entity)
                current_entity = entity

        merged.append(current_entity)
        return merged

    async def process_text_advanced(self, query: str, context: Optional[List[str]] = None, ner_confidence_threshold: float = NER_CONFIDENCE_THRESHOLD) -> Dict[str, any]:
        logger.info("Starting advanced text processing")
        resolved_query = self.resolve_coreferences(query)
        logger.debug(f"Resolved query: {resolved_query}")

        # Start the asynchronous tasks
        try:
            entities_task = asyncio.create_task(asyncio.to_thread(self.process_ner, resolved_query, ner_confidence_threshold))
            relations_task = asyncio.create_task(asyncio.to_thread(self.extract_relations, resolved_query))
            sentiment_task = asyncio.create_task(asyncio.to_thread(self.analyze_sentiment, resolved_query))

            logger.debug("Waiting for NER processing to complete")
            entities = await asyncio.wait_for(entities_task, timeout=10)
            logger.debug(f"Entities: {entities}")

            logger.debug("Waiting for relation extraction to complete")
            relations = await asyncio.wait_for(relations_task, timeout=10)
            logger.debug(f"Relations: {relations}")

            logger.debug("Waiting for sentiment analysis to complete")
            sentiment = await asyncio.wait_for(sentiment_task, timeout=10)
            logger.debug(f"Sentiment: {sentiment}")

        except asyncio.TimeoutError:
            logger.error("A task in advanced text processing timed out")
            return {"error": "Processing timed out"}

        self.load_spacy_model()
        pos_tags = [(token.text, token.pos_) for token in self.nlp(resolved_query)]
        logger.debug(f"POS tags: {pos_tags}")

        # Handle entity linking
        logger.debug("Starting entity linking")
        try:
            linked_entities = await asyncio.wait_for(asyncio.to_thread(self.entity_linking, entities, resolved_query), timeout=10)
            logger.debug(f"Linked entities: {linked_entities}")
        except asyncio.TimeoutError:
            logger.error("Entity linking timed out")
            linked_entities = []

        result = {
            "original_query": query,
            "resolved_query": resolved_query,
            "entities": linked_entities,
            "pos_tags": pos_tags,
            "relations": relations,
            "sentiment": sentiment
        }

        if context:
            try:
                logger.debug("Processing context for NER")
                context_entities = await asyncio.wait_for(asyncio.to_thread(self.batch_process_ner, context, ner_confidence_threshold), timeout=15)
                logger.debug(f"Context entities: {context_entities}")

                logger.debug("Finding similar entities")
                similar_contexts = await asyncio.wait_for(asyncio.to_thread(self.find_similar_entities, linked_entities, context_entities), timeout=15)
                logger.debug(f"Similar contexts: {similar_contexts}")

                result['similar_contexts'] = similar_contexts[:5]
            except asyncio.TimeoutError:
                logger.error("Processing context or finding similar entities timed out")

        logger.info("Completed advanced text processing")
        return result

    @lru_cache(maxsize=1000)
    def process_ner(self, query: str, ner_confidence_threshold: float = NER_CONFIDENCE_THRESHOLD) -> List[Dict[str, str]]:
        self.load_ner_model()
        dataset = NERDataset([query], self.ner_tokenizer)
        dataloader = DataLoader(dataset, batch_size=1)
        batch = next(iter(dataloader))
        entities = self.process_ner_batch(batch, ner_confidence_threshold)[0]
        return entities

    def batch_process_ner(self, queries: List[str], ner_confidence_threshold: float = NER_CONFIDENCE_THRESHOLD) -> List[List[Dict[str, str]]]:
        self.load_ner_model()
        dataset = NERDataset(queries, self.ner_tokenizer)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        all_entities = []
        for batch in dataloader:
            batch_entities = self.process_ner_batch(batch, ner_confidence_threshold)
            all_entities.extend(batch_entities)
        return all_entities

    @lru_cache(maxsize=1000)
    def extract_relations(self, query: str) -> List[Tuple[str, str, str]]:
        self.load_spacy_model()
        doc = self.nlp(query)
        relations = []
        for sent in doc.sents:
            root = sent.root
            subject = next((child for child in root.children if child.dep_ in ("nsubj", "nsubjpass")), None)
            if subject:
                for child in root.children:
                    if child.dep_ in ("dobj", "attr"):
                        relations.append((subject.text, root.text, child.text))
        return relations

    @lru_cache(maxsize=1000)
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        self.load_sentiment_model()
        inputs = self.sentiment_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.sentiment_model.generate(
                **inputs,
                num_beams=self.num_beams,
                max_length=50,
                early_stopping=True
            )

        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = {
            "positive": scores[0][1].item(),
            "negative": scores[0][0].item()
        }
        return sentiment

    @lru_cache(maxsize=1000)
    def entity_linking(self, entities, query: str):
        # Entity linking logic here...
        linked_entities = []  # Placeholder for actual entity linking logic
        return linked_entities

    def resolve_coreferences(self, text: str) -> str:
        # Implement coreference resolution logic here
        return text  # Placeholder, return original text for now

    def find_similar_entities(self, entities, context_entities):
        # Implement entity similarity logic here
        return []  # Placeholder, return empty list for now

async def main():
    # Example usage of the NERProcessor class
    ner_processor = NERProcessor(
        ner_model_name=ner_model_name,
        sentiment_model_name=sentiment_model_name,
        flair_model_name=flair_model_name,
        spacy_model_name=spacy_model_name,
        num_beams=NUM_BEAMS
    )

    # Example usage
    query = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    result = await ner_processor.process_text_advanced(query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

# Export necessary components
__all__ = ['NERProcessor', 'ner_model_name', 'sentiment_model_name', 'flair_model_name', 'spacy_model_name']