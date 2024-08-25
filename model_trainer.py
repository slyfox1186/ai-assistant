#!/usr/bin/env python3

import time
import json
import logging
import os
import torch
import asyncio
import traceback
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, get_linear_schedule_with_warmup
from typing import List, Dict, Tuple, Union, Optional
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from config import (
    HF_CACHE_DIR, NER_MODEL_PATH, SIMILARITY_MODEL_PATH, BATCH_SIZE,
    EVAL_BATCH_SIZE, EPOCHS, LEARNING_RATE, NER_MAX_LENGTH, 
    SIMILARITY_MAX_LENGTH, NER_NUM_LABELS, MODEL_SIZE_THRESHOLD, INTERACTIONS_DIR
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAINING_DATA_FILE = os.path.join(INTERACTIONS_DIR, 'training_data.json')
MAX_EPOCH_TIME = 300  # 5 minutes
MAX_TRAINING_TIME = 1800  # 30 minutes

class TimeoutException(Exception): pass

class InteractionDataset(Dataset):
    def __init__(self, interactions, tokenizer, max_length, task):
        logger.debug(f"Initializing InteractionDataset with task: {task}")
        self.interactions = interactions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self):
        dataset_length = len(self.interactions)
        logger.debug(f"Dataset length: {dataset_length}")
        return dataset_length

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        logger.debug(f"Fetching interaction at index {idx}: {interaction}")
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

        logger.debug(f"Tokenized query: {interaction['query']} | Token IDs: {encoding['input_ids']}")

        if self.task == 'similarity':
            if 'similarity_label' in interaction:
                labels = torch.tensor(interaction['similarity_label'], dtype=torch.float)
            else:
                labels = torch.tensor(0.0, dtype=torch.float)
            logger.debug(f"Assigned similarity label: {labels}")
        else:  # 'ner'
            if 'ner_labels' in interaction:
                tokens = encoding['input_ids'].squeeze().tolist()
                try:
                    labels = validate_and_convert_ner_labels(interaction['ner_labels'], tokens, self.tokenizer)
                    if len(labels) < self.max_length:
                        labels = torch.nn.functional.pad(labels, (0, self.max_length - len(labels)), value=-100)
                    else:
                        labels = labels[:self.max_length]
                    logger.debug(f"Assigned NER labels: {labels}")
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
    logger.debug("Validating and converting NER labels")
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

    logger.debug(f"Validated NER labels: {label_list}")
    return torch.tensor(label_list, dtype=torch.long)

def load_training_data(file_path):
    logger.debug("Loading training data")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {TRAINING_DATA_FILE}: {str(e)}")
        logger.info("Attempting to fix the JSON file...")
        try:
            with open(TRAINING_DATA_FILE, 'r') as f:
                content = f.read()
            # Remove the problematic character identified
            problematic_index = e.pos
            fixed_content = content[:problematic_index] + content[problematic_index+1:]
            fixed_data = json.loads(fixed_content)
            logger.info("Successfully fixed and loaded the JSON file")
            
            # Save the fixed JSON back to the file
            with open(TRAINING_DATA_FILE, 'w') as f:
                json.dump(fixed_data, f, indent=2)
            logger.info("Fixed JSON has been saved back to the file")
            
            return fixed_data
        except Exception as fix_error:
            logger.error(f"Failed to fix the JSON file: {str(fix_error)}")
            return []
    except Exception as e:
        logger.error(f"Error loading training data from {TRAINING_DATA_FILE}: {str(e)}")
        return []

async def train_ner_model(interactions: List[Dict[str, str]], max_epochs: int) -> Tuple[Optional[AutoModelForTokenClassification], Optional[AutoTokenizer]]:
    logger.info("Starting NER model training")

    training_data = load_training_data("interactions/training_data.json")
    all_interactions = training_data + interactions
    cleaned_interactions = clean_interactions(all_interactions)
    
    valid_interactions = [i for i in cleaned_interactions if 'ner_labels' in i]
    
    logger.debug(f"Total valid interactions for NER training: {len(valid_interactions)}")

    if not valid_interactions:
        logger.warning("No valid interactions with 'ner_labels' found. Skipping NER model training.")
        return None, None

    train_data, val_data = train_test_split(valid_interactions, test_size=0.2, random_state=42)
    logger.debug(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', clean_up_tokenization_spaces=True)
    model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=NER_NUM_LABELS)

    train_dataset = InteractionDataset(train_data, tokenizer, NER_MAX_LENGTH, 'ner')
    val_dataset = InteractionDataset(val_data, tokenizer, NER_MAX_LENGTH, 'ner')
    
    num_workers = min(4, cpu_count())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(NER_MODEL_PATH, 'best_model_checkpoint.pt')
    min_delta = 1e-4  # Minimum change in validation loss to be considered as improvement

    start_time = time.time()
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        epoch_start_time = time.time()
        total_batches = len(train_dataloader)
        logger.info(f"Starting Epoch {epoch+1}/{max_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            if time.time() - epoch_start_time > MAX_EPOCH_TIME:
                logger.warning(f"Epoch {epoch+1} exceeded time limit. Moving to next epoch.")
                break

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_train_loss += loss.item()
            logger.debug(f"Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{max_epochs}, Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1}/{max_epochs}, Validation Loss: {avg_val_loss:.4f}")

        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, best_model_path)
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        if time.time() - start_time > MAX_TRAINING_TIME:
            logger.warning(f"NER model training exceeded maximum time limit of {MAX_TRAINING_TIME} seconds. Stopping training.")
            break

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best NER model from checkpoint with validation loss: {checkpoint['val_loss']:.4f}")
        return model, tokenizer
    else:
        logger.warning("No best NER model checkpoint found")
        return None, None

async def train_similarity_model(interactions: List[Dict[str, str]], max_epochs: int) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer]]:
    logger.info("Starting similarity model training")
    
    training_data = load_training_data("interactions/training_data.json")
    all_interactions = training_data + interactions
    cleaned_interactions = clean_interactions(all_interactions)
    
    valid_interactions = [i for i in cleaned_interactions if 'similarity_label' in i]
    
    logger.debug(f"Total valid interactions for similarity training: {len(valid_interactions)}")

    if not valid_interactions:
        logger.warning("No valid interactions with 'similarity_label' found. Skipping similarity model training.")
        return None, None

    train_data, val_data = train_test_split(valid_interactions, test_size=0.2, random_state=42)
    logger.debug(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

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

    start_time = time.time()
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        epoch_start_time = time.time()
        total_batches = len(train_dataloader)
        logger.info(f"Starting Epoch {epoch+1}/{max_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            if time.time() - epoch_start_time > MAX_EPOCH_TIME:
                logger.warning(f"Epoch {epoch+1} exceeded time limit. Moving to next epoch.")
                break

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                pooled_output = last_hidden_state.mean(dim=1)
                labels = labels.view(-1, 1).float()
                loss = torch.nn.functional.mse_loss(pooled_output[:, 0].unsqueeze(1), labels)

            total_train_loss += loss.item()
            logger.debug(f"Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{max_epochs}, Average Training Loss: {avg_train_loss:.4f}")

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
                val_loss += torch.nn.functional.mse_loss(pooled_output[:, 0].unsqueeze(1), labels).item()

        val_loss /= len(val_dataloader)
        logger.info(f"Epoch {epoch+1}/{max_epochs}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(SIMILARITY_MODEL_PATH, 'best_model_checkpoint.pt'))
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        if time.time() - start_time > MAX_TRAINING_TIME:
            logger.warning(f"Similarity model training exceeded maximum time limit of {MAX_TRAINING_TIME} seconds. Stopping training.")
            break

    checkpoint = torch.load(os.path.join(SIMILARITY_MODEL_PATH, 'best_model_checkpoint.pt'), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best Similarity model from checkpoint")

    return model, tokenizer

def save_models(ner_model, ner_tokenizer, similarity_model, similarity_tokenizer):
    logger.info("Saving models to disk...")
    if ner_model:
        ner_model.save_pretrained(NER_MODEL_PATH)
        ner_tokenizer.save_pretrained(NER_MODEL_PATH)
        logger.info("NER model saved successfully.")
    if similarity_model:
        similarity_model.save_pretrained(SIMILARITY_MODEL_PATH)
        similarity_tokenizer.save_pretrained(SIMILARITY_MODEL_PATH)
        logger.info("Similarity model saved successfully.")

async def train_models(progress_callback=None):
    logger.info("Starting model training...")
    try:
        # Ensure the correct file_path is passed
        file_path = "interactions/training_data.json"
        logger.info("Loading training data...")
        training_data = load_training_data(file_path)
        logger.info(f"Successfully loaded training data, total records: {len(training_data)}")
        
        max_epochs = 3  # Define the number of epochs

        # Update progress after loading data
        if progress_callback:
            progress_callback(0.1)

        logger.info("Starting NER model training...")
        ner_model, ner_tokenizer = await train_ner_model(training_data, max_epochs)
        logger.info("NER model training completed")
        
        # Update progress after NER model training
        if progress_callback:
            progress_callback(0.5)
        
        logger.info("Starting Similarity model training...")
        similarity_model, similarity_tokenizer = await train_similarity_model(training_data, max_epochs)
        logger.info("Similarity model training completed")
        
        # Update progress after Similarity model training
        if progress_callback:
            progress_callback(0.8)
        
        # Save the models if training is successful
        if ner_model and similarity_model:
            logger.info("Saving trained models...")
            save_models(ner_model, ner_tokenizer, similarity_model, similarity_tokenizer)
            logger.info("Models saved successfully")
        else:
            logger.warning("One or both models failed to train. Not saving models.")
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        return True
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def clear_dynamic_interaction_files():
    for filename in os.listdir(INTERACTIONS_DIR):
        if filename.startswith("interaction_") and filename.endswith(".json"):
            os.remove(os.path.join(INTERACTIONS_DIR, filename))
            logger.info(f"Removed dynamic interaction file: {filename}")
    logger.info("training_data.json is kept intact")

def clean_interactions(interactions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    logger.debug("Cleaning interactions")
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
    logger.debug(f"Cleaned {len(cleaned_interactions)} interactions")
    return cleaned_interactions

async def run_training(iterations, progress_callback=None):
    logger.info(f"Starting run_training with {iterations} iterations")
    try:
        if iterations > 0:
            success = await train_models(progress_callback)
            if success:
                logger.info("Training completed successfully.")
            else:
                logger.error("Training failed.")
            return success
        else:
            logger.info("No iterations specified, skipping training.")
            return False
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train NER and Similarity models")
    parser.add_argument("--iterations", type=int, default=1, help="Number of training iterations")
    args = parser.parse_args()

    logger.info("Starting main execution")
    try:
        success = asyncio.run(run_training(args.iterations))
        if success:
            logger.info("Training process exited successfully.")
        else:
            logger.error("Training process encountered an error and is exiting.")
    except Exception as e:
        logger.error(f"Error occurred during training: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Main execution finished.")
