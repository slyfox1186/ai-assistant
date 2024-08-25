#!/usr/bin/env python3

import aiofiles
import asyncio
import json
import logging
import os
import pandas as pd
import pickle
import sys
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
from config import INTERACTIONS_DIR, DATABASE_URL, MAX_WORKERS, CHUNK_SIZE, NER_CONFIDENCE_THRESHOLD, NUM_BEAMS
from contextlib import contextmanager
from ner_processor import NERProcessor, ner_model_name, sentiment_model_name, flair_model_name, spacy_model_name
from similarity_utils import compute_similarity
from sqlalchemy import create_engine, Column, Integer, String, JSON, LargeBinary, Float
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from typing import List, Dict, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True)
    query = Column(String(512))
    answer = Column(String(1024))
    ner_labels = Column(JSON)
    similarity_label = Column(Float)

class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True)
    model_state = Column(LargeBinary)
    tokenizer_state = Column(LargeBinary)

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
Base.metadata.create_all(engine)
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)

@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error in database session: {e}")
        raise
    finally:
        session.close()

class DataManager:
    def __init__(self):
        os.makedirs(INTERACTIONS_DIR, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.semaphore = Semaphore(MAX_WORKERS)
        self.ner_processor = NERProcessor(
            ner_model_name=ner_model_name,
            sentiment_model_name=sentiment_model_name,
            flair_model_name=flair_model_name,
            spacy_model_name=spacy_model_name,
            num_beams=NUM_BEAMS  # Pass the num_beams configuration here
        )

    async def save_interaction(self, query: str, answer: str, meta_data: Optional[Dict] = None) -> None:
        async with self.semaphore:
            ner_results = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self.ner_processor.process_ner, 
                query, 
                NER_CONFIDENCE_THRESHOLD
            )
            
            tokens = query.split()
            ner_labels = [0] * len(tokens)
            for ner in ner_results:
                for i, token in enumerate(tokens):
                    if ner['word'].strip() == token.strip():
                        ner_labels[i] = 1
                        break
            
            similarity_label = await asyncio.get_event_loop().run_in_executor(self.executor, compute_similarity, query, answer)
            
            interaction = {
                "query": query,
                "answer": answer,
                "ner_labels": ner_labels,
                "similarity_label": similarity_label
            }
            filename = f"{INTERACTIONS_DIR}/interaction_{len(os.listdir(INTERACTIONS_DIR)) + 1}.json"
            
            try:
                async with aiofiles.open(filename, "w") as f:
                    await f.write(json.dumps(interaction, indent=2))
                
                await asyncio.get_event_loop().run_in_executor(self.executor, self._save_to_db, interaction)
                
                logger.info(f"Interaction saved: {filename}")
            except Exception as e:
                logger.error(f"Error saving interaction: {e}")

    def _save_to_db(self, interaction):
        with session_scope() as session:
            db_interaction = Interaction(**interaction)
            session.add(db_interaction)

    async def load_interactions(self) -> List[Dict[str, Union[str, Dict]]]:
        async with self.semaphore:
            file_interactions = await self._load_from_files()
            db_interactions = await asyncio.get_event_loop().run_in_executor(self.executor, self._load_from_db)
            return file_interactions + db_interactions

    async def _load_from_files(self) -> List[Dict[str, Union[str, Dict]]]:
        interactions = []
        try:
            filenames = [f for f in os.listdir(INTERACTIONS_DIR) if f.startswith("interaction_") and f.endswith(".json")]
            
            async def read_file(filename):
                async with aiofiles.open(f"{INTERACTIONS_DIR}/{filename}", "r") as f:
                    return json.loads(await f.read())
            
            tasks = [asyncio.create_task(read_file(filename)) for filename in filenames]
            interactions = await asyncio.gather(*tasks)
            
            logger.info(f"Loaded {len(interactions)} interactions from files")
        except Exception as e:
            logger.error(f"Error loading interactions from files: {e}")
        return interactions

    def _load_from_db(self) -> List[Dict[str, Union[str, Dict]]]:
        with session_scope() as session:
            db_interactions = session.query(Interaction).all()
            interactions = [
                {
                    "query": i.query,
                    "answer": i.answer,
                    "ner_labels": i.ner_labels,
                    "similarity_label": i.similarity_label
                }
                for i in db_interactions
            ]
            logger.info(f"Loaded {len(interactions)} interactions from database")
            return interactions

    async def batch_save_interactions(self, interactions: List[Dict[str, Union[str, Dict]]]) -> None:
        save_tasks = [self.save_interaction(i['query'], i['answer']) for i in interactions]
        await asyncio.gather(*save_tasks)

    async def export_to_csv(self, filename: str) -> None:
        interactions = await self.load_interactions()
        df = pd.DataFrame(interactions)
        df.to_csv(filename, index=False)
        logger.info(f"Interactions exported to CSV: {filename}")

    async def import_from_csv(self, filename: str) -> None:
        df = pd.read_csv(filename)
        interactions = df.to_dict('records')
        await self.batch_save_interactions(interactions)
        logger.info(f"Interactions imported from CSV: {filename}")

    def delete_interaction(self, interaction_id: int) -> None:
        try:
            with session_scope() as session:
                interaction = session.query(Interaction).filter_by(id=interaction_id).first()
                if interaction:
                    session.delete(interaction)
                    logger.info(f"Interaction {interaction_id} deleted from database")
                else:
                    logger.warning(f"Interaction {interaction_id} not found in database")
        except Exception as e:
            logger.error(f"Error deleting interaction: {e}")

    def update_interaction(self, interaction_id: int, query: Optional[str] = None, 
                           answer: Optional[str] = None) -> None:
        try:
            with session_scope() as session:
                interaction = session.query(Interaction).filter_by(id=interaction_id).first()
                if interaction:
                    if query:
                        interaction.query = query
                    if answer:
                        interaction.answer = answer
                    if query or answer:
                        ner_results = self.ner_processor.process_ner(query or interaction.query, NER_CONFIDENCE_THRESHOLD)
                        interaction.ner_labels = [0 if ner['entity'] == 'O' else 1 for ner in ner_results]
                        interaction.similarity_label = compute_similarity(query or interaction.query, answer or interaction.answer)
                    logger.info(f"Interaction {interaction_id} updated in database")
                else:
                    logger.warning(f"Interaction {interaction_id} not found in database")
        except Exception as e:
            logger.error(f"Error updating interaction: {e}")

    def search_interactions(self, query: str) -> List[Dict[str, Union[str, Dict]]]:
        try:
            with session_scope() as session:
                results = session.query(Interaction).filter(Interaction.query.ilike(f"%{query}%")).all()
                return [
                    {
                        "id": r.id,
                        "query": r.query,
                        "answer": r.answer,
                        "ner_labels": r.ner_labels,
                        "similarity_label": r.similarity_label
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Error searching interactions: {e}")
            return []

    def get_interaction_count(self) -> int:
        try:
            with session_scope() as session:
                return session.query(Interaction).count()
        except Exception as e:
            logger.error(f"Error getting interaction count: {e}")
            return 0

    def clear_all_interactions(self) -> None:
        try:
            with session_scope() as session:
                session.query(Interaction).delete()
            
            for filename in os.listdir(INTERACTIONS_DIR):
                if filename.startswith("interaction_") and filename.endswith(".json"):
                    os.remove(f"{INTERACTIONS_DIR}/{filename}")
            
            logger.info("All interactions cleared from database and files")
        except Exception as e:
            logger.error(f"Error clearing all interactions: {e}")

    async def process_interactions_in_chunks(self, processor_func, chunk_size: int = CHUNK_SIZE):
        interactions = await self.load_interactions()
        for i in range(0, len(interactions), chunk_size):
            chunk = interactions[i:i + chunk_size]
            await processor_func(chunk)

    def save_model_to_database(self, model_name: str, model_state: dict, tokenizer_state: dict):
        with session_scope() as session:
            existing_model = session.query(Model).filter_by(name=model_name).first()
            
            model_state_serialized = pickle.dumps(model_state)
            tokenizer_state_serialized = pickle.dumps(tokenizer_state)
            
            if existing_model:
                existing_model.model_state = model_state_serialized
                existing_model.tokenizer_state = tokenizer_state_serialized
                logger.info(f"Model {model_name} updated in the database.")
            else:
                new_model = Model(
                    name=model_name,
                    model_state=model_state_serialized,
                    tokenizer_state=tokenizer_state_serialized
                )
                session.add(new_model)
                logger.info(f"Model {model_name} saved to the database.")

    def load_model_from_database(self, model_name: str):
        try:
            with session_scope() as session:
                model_data = session.query(Model).filter_by(name=model_name).first()
                if model_data:
                    return {
                        'model_state': pickle.loads(model_data.model_state),
                        'tokenizer_state': pickle.loads(model_data.tokenizer_state)
                    }
                else:
                    logger.info(f"Model {model_name} not found in database")
                    return None
        except Exception as e:
            logger.error(f"Error loading model {model_name} from database: {e}")
            return None

    async def convert_existing_interactions(self):
        for filename in os.listdir(INTERACTIONS_DIR):
            if filename.startswith("interaction_") and filename.endswith(".json"):
                filepath = os.path.join(INTERACTIONS_DIR, filename)
                try:
                    async with aiofiles.open(filepath, "r") as f:
                        data = json.loads(await f.read())
                    
                    new_data = {
                        "query": data["query"],
                        "answer": data["answer"],
                        "ner_labels": data.get("ner_labels", []),
                        "similarity_label": data.get("similarity_label", 0.0)
                    }
                    
                    async with aiofiles.open(filepath, "w") as f:
                        await f.write(json.dumps(new_data, indent=2))
                    
                    logger.info(f"Converted {filename} to new format")
                except Exception as e:
                    logger.error(f"Error converting {filename}: {e}")

    def clear_interaction_files(self) -> None:
        try:
            for filename in os.listdir(INTERACTIONS_DIR):
                if filename.startswith("interaction_") and filename.endswith(".json"):
                    os.remove(os.path.join(INTERACTIONS_DIR, filename))
            logger.info("Interaction files cleared.")
        except Exception as e:
            logger.error(f"Error clearing interaction files: {str(e)}")

data_manager = DataManager()

# Exposed asynchronous functions
async def save_interaction(query: str, answer: str) -> None:
    await data_manager.save_interaction(query, answer)

async def load_interactions() -> List[Dict[str, Union[str, Dict]]]:
    return await data_manager.load_interactions()

async def batch_save_interactions(interactions: List[Dict[str, Union[str, Dict]]]) -> None:
    await data_manager.batch_save_interactions(interactions)

async def export_to_csv(filename: str) -> None:
    await data_manager.export_to_csv(filename)

async def import_from_csv(filename: str) -> None:
    await data_manager.import_from_csv(filename)

async def process_interactions_in_chunks(processor_func, chunk_size: int = CHUNK_SIZE):
    await data_manager.process_interactions_in_chunks(processor_func, chunk_size)

async def convert_existing_interactions():
    await data_manager.convert_existing_interactions()

# Exposed synchronous functions
def delete_interaction(interaction_id: int) -> None:
    data_manager.delete_interaction(interaction_id)

def update_interaction(interaction_id: int, query: Optional[str] = None, 
                       answer: Optional[str] = None) -> None:
    data_manager.update_interaction(interaction_id, query, answer)

def search_interactions(query: str) -> List[Dict[str, Union[str, Dict]]]:
    return data_manager.search_interactions(query)

def get_interaction_count() -> int:
    return data_manager.get_interaction_count()

def clear_all_interactions() -> None:
    data_manager.clear_all_interactions()

def save_model_to_database(model_name: str, model_state: dict, tokenizer_state: dict):
    data_manager.save_model_to_database(model_name, model_state, tokenizer_state)

def load_model_from_database(model_name: str):
    return data_manager.load_model_from_database(model_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <query> [<answer>]")
        sys.exit(1)
    
    query = sys.argv[1]
    answer = sys.argv[2] if len(sys.argv) > 2 else None

    if answer:
        save_interaction(query, answer)
        print(f"Interaction saved: Query: '{query}' -> Answer: '{answer}'")
    else:
        print(f"Only query provided: '{query}'. No answer to save.")

    interactions = load_interactions()
    print(f"Number of interactions: {len(interactions)}")

    dummy_model_state = {"layers": [1, 2, 3]}
    dummy_tokenizer_state = {"vocab": ["a", "b", "c"]}
    
    model_name = "dummy_model"
    save_model_to_database(model_name, dummy_model_state, dummy_tokenizer_state)
    print(f"Model '{model_name}' saved to database.")

    loaded_model = load_model_from_database(model_name)
    if loaded_model:
        print(f"Loaded model: {loaded_model}")
    else:
        print(f"Failed to load model '{model_name}'.")

    # Convert existing interactions to the new format
    convert_existing_interactions()
    print("Existing interactions converted to new format.")

# Export the necessary functions
__all__ = [
    'save_interaction', 
    'load_interactions', 
    'batch_save_interactions', 
    'export_to_csv',
    'import_from_csv',
    'delete_interaction',
    'update_interaction',
    'search_interactions',
    'get_interaction_count',
    'clear_all_interactions',
    'process_interactions_in_chunks',
    'save_model_to_database',
    'load_model_from_database',
    'convert_existing_interactions'
]
