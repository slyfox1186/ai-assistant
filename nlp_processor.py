#!/usr/bin/env python3

import logging
import torch
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from functools import lru_cache
import nltk
import os
from diskcache import Cache

logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

cache = Cache('model_cache')

@lru_cache(maxsize=2)
def load_model(model_name):
    device = get_device()
    model, tokenizer = cache.get(model_name, (None, None))
    if model is None or tokenizer is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='model_cache').to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='model_cache')
        cache.set(model_name, (model, tokenizer))
    return model, tokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RetrievalAugmentedGeneration:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=1.0)
        self.texts = []

    def add_to_knowledge_base(self, texts):
        self.texts.extend(texts)
        if self.texts:
            try:
                self.vectorizer.fit(self.texts)
            except ValueError as e:
                logger.warning(f"Error fitting vectorizer: {e}")
                # If fitting fails, try with more lenient parameters
                self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=1.0)
                self.vectorizer.fit(self.texts)

    def retrieve(self, query, k=5):
        if not self.texts:
            return []
        query_vector = self.vectorizer.transform([query])
        text_vectors = self.vectorizer.transform(self.texts)
        similarities = cosine_similarity(query_vector, text_vectors)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.texts[i] for i in top_k_indices]

class NLPProcessor:
    def __init__(self):
        self.device = get_device()
        self.model_name = "facebook/bart-large-cnn"
        self.model, self.tokenizer = load_model(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.rag = RetrievalAugmentedGeneration()
        self.exclusion_criteria = self.load_exclusion_criteria()

    def load_exclusion_criteria(self):
        criteria_path = 'exclusion_criteria.json'
        if os.path.exists(criteria_path):
            with open(criteria_path, 'r') as file:
                return json.load(file).get('phrases_to_exclude', [])
        return []

    def update_exclusion_criteria(self):
        self.exclusion_criteria = self.load_exclusion_criteria()

    def clean_response(self, response):
        sentences = sent_tokenize(response)
        cleaned_sentences = [
            sent for sent in sentences
            if not any(phrase in sent.lower() for phrase in self.exclusion_criteria)
        ]
        return " ".join(cleaned_sentences)

    def generate_response(self, input_text, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2):
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=no_repeat_ngram_size,
                attention_mask=attention_mask,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.exception(f"Error in generate_response: {e}")
            return "An error occurred while generating the response."

    def retrieve_relevant_info(self, query, content_list):
        try:
            self.rag.add_to_knowledge_base(content_list)
            retrieved_info = self.rag.retrieve(query)
            logger.info(f"Number of retrieved info items: {len(retrieved_info)}")
            context = " ".join([str(info) for info in retrieved_info])
            if not context:
                context = "No relevant information found."
            return context
        except Exception as e:
            logger.error(f"Error in retrieve_relevant_info: {e}")
            return "Error retrieving relevant information."

    def process_query(self, query, content_list, arg1, arg2):
        context = self.retrieve_relevant_info(query, content_list)
        input_text = f"Answer the following question based on the given context. {arg1} {arg2}\nQuestion: {query}\nContext: {context}"
        response = self.generate_response(input_text)
        self.update_exclusion_criteria()
        cleaned_response = self.clean_response(response)
        return cleaned_response

    def extract_key_information(self, text):
        sentences = sent_tokenize(text)
        if not sentences:
            return "No key information extracted."
        
        stop_words = set(stopwords.words('english'))
        filtered_sentences = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in sentences]
        
        vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
        tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        def get_top_keywords(tfidf_row, feature_names, top_n=5):
            sorted_items = sorted(zip(tfidf_row, feature_names), key=lambda x: x[0], reverse=True)
            return [item[1] for item in sorted_items[:top_n]]
        
        keywords = [get_top_keywords(tfidf_matrix[i].toarray()[0], feature_names) for i in range(len(sentences))]
        
        flat_keywords = [keyword for sublist in keywords for keyword in sublist]
        
        sentence_scores = [sum([1 for keyword in sentence_keywords if keyword in flat_keywords]) for sentence_keywords in keywords]
        
        top_sentence_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:3]
        key_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
        
        return ' '.join(key_sentences) if key_sentences else "No key information extracted."

    def process_nlp(self, data, query, arg1, arg2):
        logger.info("Starting NLP processing...")
        try:
            content_list = [item['content'] for item in data if isinstance(item.get('content'), str) and item['content'].strip()]
            logger.info(f"Number of content items: {len(content_list)}")
            
            cleaned_response = self.process_query(query, content_list, arg1, arg2)
            
            key_info = self.extract_key_information(cleaned_response)
            
            processed_response = {
                'query': query,
                'response': cleaned_response,
                'key_info': key_info
            }
            
            logger.info("NLP processing completed.")
            return processed_response
        except Exception as e:
            logger.exception(f"Error during NLP processing: {e}")
            return {
                'query': query,
                'response': "An error occurred during processing. Please try again.",
                'key_info': "No key information available due to processing error."
            }

    def train_model(self, data, query, response, num_epochs=3):
        input_texts = [item['content'] for item in data if isinstance(item.get('content'), str) and item['content'].strip()]
        input_text = " ".join(input_texts) + f" {query}"
        
        logger.info(f"Training the model with query: {query}")
        logger.info(f"Response: {response}")
        
        train_dataset = self.create_train_dataset(input_text, response)
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        logger.info("Model training completed.")

    def create_train_dataset(self, input_text, response):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        labels = self.tokenizer.encode(response, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return torch.utils.data.TensorDataset(input_ids, labels)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP Processor Script")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to the data file")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to process")
    parser.add_argument("--arg1", type=str, required=False, help="First argument to modify the query processing")
    parser.add_argument("--arg2", type=str, required=False, help="Second argument to modify the query processing")
    args = parser.parse_args()

    with open(args.data, 'r') as file:
        data = json.load(file)

    nlp_processor = NLPProcessor()
    try:
        result = nlp_processor.process_nlp(data, args.query, args.arg1, args.arg2)
        print(json.dumps(result, indent=4))
    except Exception as e:
        logger.error(f"Failed to process NLP task: {e}")