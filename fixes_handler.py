#!/usr/bin/env python3
import torch
import sqlite3
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

class FixesHandler:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_model.to(self.device)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.gpt2_model.to(self.device)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model=self.t5_model, tokenizer=self.t5_tokenizer, device=0 if torch.cuda.is_available() else -1)

    def enhance_answer(self, answer):
        inputs = self.t5_tokenizer.encode(f"enhance: {answer}", return_tensors='pt', max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        outputs = self.t5_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        enhanced_answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return enhanced_answer

    def train_adaptive_model(self, query, response, feedback_score):
        inputs = self.gpt2_tokenizer(query + self.gpt2_tokenizer.sep_token + response, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        labels = torch.tensor([feedback_score]).unsqueeze(0).to(self.device)
        
        self.gpt2_model.train()
        outputs = self.gpt2_model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        optimizer = torch.optim.AdamW(self.gpt2_model.parameters(), lr=5e-5)
        optimizer.step()
        optimizer.zero_grad()

    def analyze_data(self, data):
        embeddings = self.sentence_transformer.encode(data, convert_to_tensor=True, device=self.device)
        return embeddings

    def summarize_content(self, data):
        summaries = []
        for item in data:
            summary = self.summarizer(item['content'], max_length=min(len(item['content']) // 2, 256), min_length=10, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return summaries

    def fetch_response(self, db_path, query):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM queries WHERE query=?", (query,))
        response = cursor.fetchone()
        conn.close()
        return response

    def insert_query(self, db_path, query, response):
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute("INSERT INTO queries (query, response) VALUES (?, ?)", (query, response))
        conn.close()
