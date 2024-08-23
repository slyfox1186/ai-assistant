#!/usr/bin/env python3

import torch
import spacy
import numpy as np
import nltk
import logging
import json
import asyncio
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from flair.models import SequenceTagger
from flair.data import Sentence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model names
ner_model_name = "jean-baptiste/roberta-large-ner-english"
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
flair_model_name = "flair/ner-english-ontonotes-large"
spacy_model_name = "en_core_web_trf"

class NERDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
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
    def __init__(self, ner_model_name: str, sentiment_model_name: str, flair_model_name: str, spacy_model_name: str):
        # Load models dynamically based on user input or configuration
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)

        self.nlp = spacy.load(spacy_model_name)
        self.flair_ner = SequenceTagger.load(flair_model_name)

        # Initialize caches
        self.ner_cache = {}
        self.relation_cache = {}
        self.sentiment_cache = {}
        self.entity_linking_cache = {}

        self.lemmatizer = nltk.WordNetLemmatizer()

    def process_ner_batch(self, batch: Dict[str, torch.Tensor], ner_confidence_threshold: float) -> List[List[Dict[str, str]]]:
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

    @lru_cache(maxsize=1000)
    def process_ner(self, query: str, ner_confidence_threshold: float) -> List[Dict[str, str]]:
        dataset = NERDataset([query], self.ner_tokenizer)
        dataloader = DataLoader(dataset, batch_size=1)
        batch = next(iter(dataloader))
        return self.process_ner_batch(batch, ner_confidence_threshold)[0]

    @lru_cache(maxsize=1000)
    def extract_relations(self, doc: Doc) -> List[Tuple[str, str, str]]:
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
        inputs = self.sentiment_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)

        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "positive": scores[0][1].item(),
            "negative": scores[0][0].item()
        }

    @lru_cache(maxsize=1000)
    def entity_linking(self, entities: List[Dict[str, str]], text: str) -> List[Dict[str, any]]:
        sentence = Sentence(text)
        self.flair_ner.predict(sentence)

        linked_entities = []
        for entity in entities:
            flair_entity = next((e for e in sentence.get_spans('ner') if e.text == entity['word']), None)
            if flair_entity:
                synsets = nltk.corpus.wordnet.synsets(self.lemmatizer.lemmatize(entity['word']))
                linked_entity = {
                    **entity,
                    'flair_label': flair_entity.labels[0].value,
                    'flair_score': flair_entity.score,
                    'synsets': [{'name': syn.name(), 'definition': syn.definition()} for syn in synsets[:3]]
                }
                linked_entities.append(linked_entity)
            else:
                linked_entities.append(entity)

        return linked_entities

    def resolve_coreferences(self, text: str) -> str:
        doc = self.nlp(text)
        resolved_text = text
        for cluster in doc._.coref_clusters:
            for mention in cluster.mentions[1:]:
                resolved_text = resolved_text.replace(mention.text, cluster.main.text)
        return resolved_text

    async def process_text_advanced(self, query: str, context: Optional[List[str]] = None, ner_confidence_threshold: float = 0.9) -> Dict[str, any]:
        resolved_query = self.resolve_coreferences(query)
        doc = self.nlp(resolved_query)

        entities_task = asyncio.create_task(asyncio.to_thread(self.process_ner, resolved_query, ner_confidence_threshold))
        relations_task = asyncio.create_task(asyncio.to_thread(self.extract_relations, doc))
        sentiment_task = asyncio.create_task(asyncio.to_thread(self.analyze_sentiment, resolved_query))

        entities = await entities_task
        relations = await relations_task
        sentiment = await sentiment_task

        pos_tags = [(token.text, token.pos_) for token in doc]
        linked_entities = await asyncio.to_thread(self.entity_linking, entities, resolved_query)

        result = {
            "original_query": query,
            "resolved_query": resolved_query,
            "entities": linked_entities,
            "pos_tags": pos_tags,
            "relations": relations,
            "sentiment": sentiment
        }

        if context:
            context_entities = await asyncio.to_thread(self.batch_process_ner, context, ner_confidence_threshold)
            similar_contexts = await asyncio.to_thread(self.find_similar_entities, linked_entities, context_entities)
            result['similar_contexts'] = similar_contexts[:5]

        return result

    def batch_process_ner(self, queries: List[str], ner_confidence_threshold: float) -> List[List[Dict[str, str]]]:
        dataset = NERDataset(queries, self.ner_tokenizer)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

        all_entities = []
        for batch in dataloader:
            batch_entities = self.process_ner_batch(batch, ner_confidence_threshold)
            all_entities.extend(batch_entities)

        return all_entities

    def clear_caches(self):
        self.ner_cache.clear()
        self.relation_cache.clear()
        self.sentiment_cache.clear()
        self.entity_linking_cache.clear()
        self.process_ner.cache_clear()
        self.extract_relations.cache_clear()
        self.analyze_sentiment.cache_clear()
        self.entity_linking.cache_clear()
        torch.cuda.empty_cache()

    def get_entity_embeddings(self, entities: List[Dict[str, str]]) -> np.ndarray:
        entity_texts = [entity['word'] for entity in entities]
        docs = list(self.nlp.pipe(entity_texts))
        return np.array([doc.vector for doc in docs])

    def find_similar_entities(self, query_entities: List[Dict[str, str]], all_entities: List[List[Dict[str, str]]]) -> List[Tuple[int, float]]:
        query_embeddings = self.get_entity_embeddings(query_entities)
        all_embeddings = [self.get_entity_embeddings(entities) for entities in all_entities]

        similarities = []
        for i, embeddings in enumerate(all_embeddings):
            if embeddings.size == 0 or query_embeddings.size == 0:
                similarities.append((i, 0))
            else:
                sim = cosine_similarity(query_embeddings, embeddings).max()
                similarities.append((i, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

async def main():
    # Example of how to use the NERProcessor class
    ner_processor = NERProcessor(ner_model_name, sentiment_model_name, flair_model_name, spacy_model_name)

    query = "Apple Inc. was founded by Steve Jobs in Cupertino, California. He was known for his innovative products."
    context = [
        "Microsoft was founded by Bill Gates in Albuquerque, New Mexico.",
        "Amazon was started by Jeff Bezos in Bellevue, Washington.",
        "Google was founded by Larry Page and Sergey Brin at Stanford University in California."
    ]

    result = await ner_processor.process_text_advanced(query, context)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

# Export necessary components
__all__ = ['NERProcessor', 'ner_model_name', 'sentiment_model_name', 'flair_model_name', 'spacy_model_name']