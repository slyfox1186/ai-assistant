#!/usr/bin/env python3

import asyncio
import glob
import logging
import os
import streamlit as st
import threading
import time
import torch
import traceback

from config import OPENAI_API_KEY as DEFAULT_API_KEY
from data_manager import save_interaction, load_interactions, clear_all_interactions
from data_manager import save_interaction, load_interactions, clear_all_interactions
from model_trainer import train_models
from ner_processor import NERProcessor
from openai_handler import get_openai_response, set_openai_api_key
from similarity_checker import find_similar_interaction
from web_scraper import WebScraper

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler to log to a file
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Hardcoded OpenAI API Key
OPENAI_API_KEY = 'ENTER YOUR OPENAI API KEY HERE OR ENTER IT IN THE GUI BEFORE EXECUTING A SEARCH'
set_openai_api_key(OPENAI_API_KEY)

def delete_interaction_files():
    logger.debug("Entering delete_interaction_files()")
    interaction_files = glob.glob(os.path.join("interactions", "interaction_*.json"))
    for file in interaction_files:
        if os.path.basename(file) != "training_data.json":
            try:
                os.remove(file)
                logger.info(f"Deleted interaction file: {file}")
            except Exception as e:
                logger.error(f"Error deleting file {file}: {str(e)}")
    logger.debug("Exiting delete_interaction_files()")

def train_and_clear(train_iterations):
    logger.debug(f"Entering train_and_clear() with {train_iterations} iterations")
    try:
        logger.info("Starting model training and data clearing process")
        train_models(train_iterations)
        delete_interaction_files()
        clear_all_interactions()
        logger.info("Training completed and interaction files deleted")
        return True
    except Exception as e:
        logger.error(f"Error during training or file deletion: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    finally:
        logger.debug("Exiting train_and_clear()")

def run_training(iterations):
    logger.debug(f"Running training process in thread with {iterations} iterations")
    success = train_models(iterations)
    if success:
        st.session_state.training_completed = True
        st.sidebar.success("Training completed!")
    else:
        st.sidebar.error("Training failed.")
    st.session_state.is_training = False
    st.session_state.train_models_toggle = False  # Ensure toggle is disabled after training

async def search_internet_and_process(query):
    scraper = WebScraper(api_key=OPENAI_API_KEY)
    comprehensive_answer = await scraper.scrape(query)
    logger.debug(f"Scraped data: {comprehensive_answer}")
    return comprehensive_answer

def main():
    logger.debug("Entering main()")
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    st.title("🤖 AI Assistant")

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    use_local_cache = st.sidebar.checkbox("Use local cached answers only")

    # Initialize session state variables
    if 'train_models_toggle' not in st.session_state:
        st.session_state.train_models_toggle = False
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'train_iterations' not in st.session_state:
        st.session_state.train_iterations = 1

    logger.debug(f"Current session state: {st.session_state}")

    # Train models toggle and iterations input
    train_models_toggle = st.sidebar.checkbox(
        "Train the models",
        value=st.session_state.train_models_toggle and not st.session_state.training_completed,
        key="train_models_checkbox",
        disabled=st.session_state.is_training
    )

    train_iterations = st.sidebar.number_input(
        "Training Iterations",
        min_value=1,
        value=st.session_state.train_iterations,
        step=1,
        disabled=st.session_state.is_training
    )
    st.session_state.train_iterations = train_iterations

    # Handle training process
    if train_models_toggle and not st.session_state.is_training:
        st.session_state.is_training = True
        st.session_state.train_models_toggle = False  # Disable toggle to prevent loop
        logger.debug("Starting training process")
        progress_bar = st.sidebar.progress(0)

        # Get the number of iterations from the session state
        iterations = st.session_state.train_iterations

        # Run training in a separate thread
        thread = threading.Thread(target=run_training, args=(iterations,))
        thread.start()

        # Show progress while training
        while thread.is_alive():
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)

        thread.join()

        logger.debug("Training process completed")
        st.sidebar.success("Training completed!")
        st.rerun()

    # Rest of the main function (chat interface, etc.)
    col1, col2 = st.sidebar.columns(2)

    if 'use_openai' not in st.session_state:
        st.session_state.use_openai = True
    if 'search_internet' not in st.session_state:
        st.session_state.search_internet = False

    def toggle_openai():
        st.session_state.use_openai = not st.session_state.use_openai
        if st.session_state.use_openai:
            st.session_state.search_internet = False

    def toggle_internet():
        st.session_state.search_internet = not st.session_state.search_internet
        if st.session_state.search_internet:
            st.session_state.use_openai = False

    use_openai = col1.checkbox("Use OpenAI", value=st.session_state.use_openai, on_change=toggle_openai)
    search_internet = col2.checkbox("Search Internet", value=st.session_state.search_internet, on_change=toggle_internet)

    # OpenAI API Key input
    openai_api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
    if openai_api_key:
        set_openai_api_key(openai_api_key)
    else:
        set_openai_api_key(DEFAULT_API_KEY)

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        logger.debug(f"Received user prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    logger.debug(f"Processing user query: {prompt}")

                    if use_local_cache:
                        interactions = load_interactions()
                        logger.debug(f"Loaded interactions: {interactions}")
                        similar_interaction, similarity_score = find_similar_interaction(prompt, interactions)
                        if similar_interaction:
                            response = "[Cached]\n" + similar_interaction["answer"]
                            logger.debug(f"Found similar interaction in cache: {response}")
                        else:
                            response = "[Cached]\nI don't have a cached answer for this query."
                            logger.debug("No cached answer found")
                    elif search_internet:
                        scraped_data = asyncio.run(search_internet_and_process(prompt))
                        logger.debug(f"Internet search result: {scraped_data}")
                        if scraped_data:
                            if use_openai:
                                openai_response = asyncio.run(get_openai_response(f"Based on the following information, answer the question: {prompt}\n\nInformation:\n{scraped_data}"))
                                response = "[OpenAI]\n" + openai_response
                            else:
                                response = "[Web]\n" + scraped_data
                            logger.debug(f"Generated response: {response}")
                        else:
                            response = "[Web]\nI couldn't find any relevant information on the internet."
                            logger.debug("No relevant information found on the internet")
                    elif use_openai:
                        openai_response = asyncio.run(get_openai_response(prompt))
                        response = "[OpenAI]\n" + openai_response
                        logger.debug(f"OpenAI response: {response}")
                    else:
                        response = "Please enable either OpenAI or Internet Search to get an answer."
                        logger.debug("No response generated: Missing OpenAI key or internet search")

                    if use_openai or search_internet:
                        save_interaction(prompt, response)
                        logger.debug("Interaction saved")

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("An error occurred while processing your query. Please try again.")

    logger.debug(f"Final session state: {st.session_state}")
    logger.debug("Exiting main()")

if __name__ == "__main__":
    logger.debug("Starting the application")
    main()