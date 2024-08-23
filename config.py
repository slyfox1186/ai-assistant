#!/usr/bin/env python3

import os

# OpenAI API Configuration
OPENAI_API_KEY = "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"

# Directory Configurations
INTERACTIONS_DIR = 'interactions'
HF_CACHE_DIR = '/home/jman/.cache/huggingface'
NER_MODEL_PATH = os.path.join(HF_CACHE_DIR, 'ner_model')
SIMILARITY_MODEL_PATH = os.path.join(HF_CACHE_DIR, 'similarity_model')

# Model Training Configurations
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5

# NER Model Configurations
NER_MAX_LENGTH = 128
NER_NUM_LABELS = 9 # Adjust based on your NER label set

# Similarity Model Configurations
SIMILARITY_MAX_LENGTH = 512

# Evaluation Configurations
SIMILARITY_THRESHOLD = 0.7
NER_CONFIDENCE_THRESHOLD = 0.9

# Set up Hugging Face cache directory
os.environ['HF_HOME'] = HF_CACHE_DIR

# Streamlit Configurations
STREAMLIT_SERVER_PORT = 8501
STREAMLIT_SERVER_HOST = "0.0.0.0"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "ai_assistant.log"

# Thread Pool Configurations
MAX_WORKERS = 4

# Model Update Configurations
UPDATE_THRESHOLD = 100  # Number of new interactions before updating models

# OpenAI API Retry Configuration
MAX_RETRIES = 3

# OpenAI API Parameters
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
ENGINE = "gpt-4o-mini"
TIMEOUT = 30

# Caching Configuration
CACHE_SIZE = 1000

# FAISS Configuration
USE_FAISS = True
INDEX_UPDATE_FREQUENCY = 100

# Database Configuration
DATABASE_URL = "sqlite:///ai_assistant.db"  # SQLite database file

# Data Processing Configuration
CHUNK_SIZE = 1000  # Size of chunks for processing large datasets

# Model Size Threshold (in parameters)
MODEL_SIZE_THRESHOLD = 100_000_000  # 100 million parameters