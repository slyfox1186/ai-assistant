#!/usr/bin/env python3

import json
import logging
import os
import torch
from sklearn.model_selection import train_test_split
from threading import Lock
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, get_linear_schedule_with_warmup
from typing import List, Dict, Tuple, Union
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from data_manager import load_interactions, save_model_to_database, load_model_from_database
from config import (HF_CACHE_DIR, NER_MODEL_PATH, SIMILARITY_MODEL_PATH, BATCH_SIZE, EVAL_BATCH_SIZE, 
                    EPOCHS, LEARNING_RATE, NER_MAX_LENGTH, SIMILARITY_MAX_LENGTH, NER_NUM_LABELS,
                    MODEL_SIZE_THRESHOLD, INTERACTIONS_DIR)
import sqlite3
import pickle
from torch.multiprocessing import cpu_count
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

training_lock = Lock()

TRAINING_DATA_FILE = os.path.join(INTERACTIONS_DIR, 'training_data.json')

class InteractionDataset(Dataset):
    def __init__(self, interactions, tokenizer, max_length, task):
        self.interactions = interactions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        encoding = self.tokenizer.encode_plus(
            interaction['query'],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.task == 'similarity':
            if 'similarity_label' in interaction:
                labels = torch.tensor(interaction['similarity_label'], dtype=torch.float)
            else:
                labels = torch.tensor(0.0, dtype=torch.float)
        else:  # 'ner'
            if 'ner_labels' in interaction:
                tokens = encoding['input_ids'].squeeze().tolist()
                try:
                    labels = validate_and_convert_ner_labels(interaction['ner_labels'], tokens, self.tokenizer)
                    if len(labels) < self.max_length:
                        labels = torch.nn.functional.pad(labels, (0, self.max_length - len(labels)), value=-100)
                    else:
                        labels = labels[:self.max_length]
                except Exception as e:
                    logger.warning(f"Error processing NER labels for interaction {idx}: {str(e)}")
                    labels = torch.full((self.max_length,), -100, dtype=torch.long)
            else:
                labels = torch.full((self.max_length,), -100, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def validate_and_convert_ner_labels(ner_labels: Union[List[Dict[str, str]], List[int]], tokens: List[int], tokenizer) -> torch.Tensor:
    if isinstance(ner_labels, list) and all(isinstance(x, int) for x in ner_labels):
        return torch.tensor(ner_labels, dtype=torch.long)
    
    label_list = []
    for token in tokens:
        if token == tokenizer.pad_token_id:
            label_list.append(-100)  # Ignore the padding token
        else:
            matched_label = 0  # Default to 'O' (outside any entity)
            word = tokenizer.decode([token]).strip()
            for ner in ner_labels:
                if isinstance(ner, dict) and 'word' in ner:
                    if ner['word'].strip() == word:
                        matched_label = 1  # Assign a positive label (1) for entities
                        break
            label_list.append(matched_label)

    return torch.tensor(label_list, dtype=torch.long)

def load_training_data():
    try:
        with open(TRAINING_DATA_FILE, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded training data from {TRAINING_DATA_FILE}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {TRAINING_DATA_FILE}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error loading training data from {TRAINING_DATA_FILE}: {str(e)}")
        return []

def train_ner_model(interactions: List[Dict[str, str]]) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    logger.info("Starting NER model training")

    training_data = load_training_data()
    all_interactions = training_data + interactions
    cleaned_interactions = clean_interactions(all_interactions)
    
    valid_interactions = [i for i in cleaned_interactions if 'ner_labels' in i]
    
    if not valid_interactions:
        logger.warning("No valid interactions with 'ner_labels' found. Skipping NER model training.")
        return None, None

    train_data, val_data = train_test_split(valid_interactions, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=NER_NUM_LABELS)

    train_dataset = InteractionDataset(train_data, tokenizer, NER_MAX_LENGTH, 'ner')
    val_dataset = InteractionDataset(val_data, tokenizer, NER_MAX_LENGTH, 'ner')
    
    num_workers = min(4, cpu_count())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                # Reshape logits and labels to compute the loss
                logits = logits.view(-1, NER_NUM_LABELS)  # Shape: (batch_size * sequence_length, num_labels)
                labels = labels.view(-1)  # Shape: (batch_size * sequence_length)

                loss = torch.nn.functional.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                
                logits = outputs.logits
                logits = logits.view(-1, NER_NUM_LABELS)
                labels = labels.view(-1)
                
                val_loss += torch.nn.functional.cross_entropy(logits, labels).item()

        val_loss /= len(val_dataloader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(NER_MODEL_PATH, 'best_model_checkpoint.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    checkpoint = torch.load(os.path.join(NER_MODEL_PATH, 'best_model_checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer

def train_similarity_model(interactions: List[Dict[str, str]]) -> Tuple[AutoModel, AutoTokenizer]:
    logger.info("Starting similarity model training")
    
    training_data = load_training_data()
    all_interactions = training_data + interactions
    cleaned_interactions = clean_interactions(all_interactions)
    
    valid_interactions = [i for i in cleaned_interactions if 'similarity_label' in i]
    
    if not valid_interactions:
        logger.warning("No valid interactions with 'similarity_label' found. Skipping similarity model training.")
        return None, None

    train_data, val_data = train_test_split(valid_interactions, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    train_dataset = InteractionDataset(train_data, tokenizer, SIMILARITY_MAX_LENGTH, 'similarity')
    val_dataset = InteractionDataset(val_data, tokenizer, SIMILARITY_MAX_LENGTH, 'similarity')
    
    num_workers = min(4, cpu_count())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
                
                # Pool the output to get a single vector per sequence
                pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

                # Adjust labels to match the pooled output shape
                labels = labels.view(-1, 1).float()  # Shape: (batch_size, 1)

                # Compute MSE loss
                loss = torch.nn.functional.mse_loss(pooled_output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                pooled_output = last_hidden_state.mean(dim=1)
                labels = labels.view(-1, 1).float()
                val_loss += torch.nn.functional.mse_loss(pooled_output, labels).item()

        val_loss /= len(val_dataloader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(SIMILARITY_MODEL_PATH, 'best_model_checkpoint.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    checkpoint = torch.load(os.path.join(SIMILARITY_MODEL_PATH, 'best_model_checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer

def train_models(iterations: int = 1):
    """
    Train both the NER and Similarity models based on user interactions.
    """
    with training_lock:
        for _ in range(iterations):
            interactions = load_interactions()

            # Train NER Model
            ner_model, ner_tokenizer = train_ner_model(interactions)
            if ner_model:
                # Save model and tokenizer states, not the entire objects
                ner_model_state = ner_model.state_dict()
                ner_tokenizer_state = ner_tokenizer.__dict__.copy()
                save_model_to_database("ner_model", ner_model_state, ner_tokenizer_state)

            # Train Similarity Model
            sim_model, sim_tokenizer = train_similarity_model(interactions)
            if sim_model:
                # Save model and tokenizer states, not the entire objects
                sim_model_state = sim_model.state_dict()
                sim_tokenizer_state = sim_tokenizer.__dict__.copy()
                save_model_to_database("similarity_model", sim_model_state, sim_tokenizer_state)

            logger.info("Model training complete.")

def clean_interactions(interactions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned_interactions = []
    for interaction in interactions:
        if 'query' in interaction and 'answer' in interaction:
            cleaned_interaction = {
                'query': interaction['query'],
                'answer': interaction['answer']
            }
            if 'ner_labels' in interaction:
                if isinstance(interaction['ner_labels'], list):
                    cleaned_interaction['ner_labels'] = interaction['ner_labels']
            if 'similarity_label' in interaction:
                if isinstance(interaction['similarity_label'], (int, float)):
                    cleaned_interaction['similarity_label'] = interaction['similarity_label']
            cleaned_interactions.append(cleaned_interaction)
    return cleaned_interactions

if __name__ == "__main__":
    train_models()
