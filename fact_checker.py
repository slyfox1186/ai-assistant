#!/usr/bin/env python3

import logging
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from database_handler import DatabaseHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactChecker:
    def __init__(self, db_handler, device=None):
        self.db_handler = db_handler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FactChecker using device: {self.device}")

        # Load pre-trained fact-checking model
        self.model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

        # Load pre-trained text correction model
        self.correction_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(self.device)
        self.correction_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def check_fact(self, claim, evidence):
        inputs = self.tokenizer(f"{claim} [SEP] {evidence}", return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        return probs[0][1].item()  # Probability of entailment

    def correct_text(self, text):
        inputs = self.correction_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.correction_model.generate(**inputs, max_length=1024, num_beams=4, early_stopping=True)
        corrected_text = self.correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def check(self, query, data):
        logger.info("Fact-checking data...")
        checked_data = []
        sources = query_wikipedia(query) + query_duckduckgo(query) + query_google(query)
        
        # If data is a string, convert it to a list with a single dictionary
        if isinstance(data, str):
            data = [{'content': data}]
        
        for item in data:
            is_misinformation = True
            for source in sources:
                score = self.check_fact(item['content'], source)
                if score > 0.5:  # Consider as entailed if score is greater than 0.5
                    is_misinformation = False
                    break
            item['is_misinformation'] = is_misinformation
            checked_data.append(item)
        
        # If we only checked one item, return a tuple instead of a list
        if len(checked_data) == 1:
            return checked_data[0]['is_misinformation'], "Fact-checking complete"
        
        return checked_data

def query_wikipedia(query):
    logger.info("Querying Wikipedia...")
    search_url = f"https://en.wikipedia.org/w/index.php?search={query}&title=Special%3ASearch&ns0=1"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all('p')
    return [p.get_text() for p in paragraphs[:5]]

def query_duckduckgo(query):
    logger.info("Querying DuckDuckGo...")
    search_url = f"https://duckduckgo.com/html/?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.find_all('a', {'class': 'result__a'})
    
    contents = []
    for result in results[:5]:
        link = result.get('href')
        page_response = requests.get(link)
        page_response.raise_for_status()
        page_soup = BeautifulSoup(page_response.content, "html.parser")
        content = page_soup.get_text()
        contents.append(content)
    return contents

def query_google(query):
    logger.info("Querying Google...")
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.find_all('div', {'class': 'BNeawe s3v9rd AP7Wnd'})
    
    contents = []
    for result in results[:5]:
        content = result.text
        contents.append(content)
    return contents

if __name__ == "__main__":
    db_handler = DatabaseHandler("ai_assistant.db")
    fact_checker = FactChecker(db_handler)
    
    user_query = input("Enter your query: ")
    user_answer = input("Enter the content to fact-check: ")
    
    fact_checked_data = fact_checker.check(user_query, [{'content': user_answer}])
    for item in fact_checked_data:
        print(f"Content: {item['content']}\nIs misinformation: {item['is_misinformation']}\n")
