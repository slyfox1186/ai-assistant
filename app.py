#!/usr/bin/env python3

import asyncio
import logging
import os
import streamlit as st
import traceback
import re
from typing import List, Dict
from streamlit.runtime.scriptrunner import add_script_run_ctx
from researcher import Researcher
from config import OPENAI_API_KEY as DEFAULT_API_KEY
from data_manager import save_interaction, load_interactions
from model_trainer import run_training as model_run_training
from openai_handler import set_openai_api_key, get_openai_response
from similarity_checker import find_similar_interaction
from web_scraper import search_internet_and_process

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler to log to a file
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Utility function to truncate text
def truncate_text(text: str, length: int = 300) -> str:
    if len(text) > length:
        return text[:length] + "..."
    return text

def filter_irrelevant_responses(response: str) -> bool:
    irrelevant_phrases = [
        "I'm sorry, but I couldn't find any relevant information for your query.",
        "Unfortunately, there are no results available for this query.",
        "No relevant information was found."
    ]
    
    for phrase in irrelevant_phrases:
        if phrase.lower() in response.lower():
            return False  # Indicates the response should be filtered out
    
    return True  # Indicates the response is relevant and should be kept

def get_fallback_response() -> str:
    return "I couldn't find any specific information, but feel free to ask another question or refine your query."

async def run_training(iterations, progress_bar):
    success = await model_run_training(iterations)
    return success

def handle_training_result(future):
    success = future.result()
    if success:
        st.session_state.training_completed = True
        st.session_state.training_success = True
    else:
        st.session_state.training_success = False
    st.session_state.is_training = False
    st.session_state.train_models_toggle = False

def format_response(response: str) -> str:
    # Format headers
    response = re.sub(r'## (.*)', r'<h2 style="color: #4CAF50; font-weight: bold;">\1</h2>', response)
    response = re.sub(r'# (.*)', r'<h1 style="color: #FFFFE0;">\1</h1>', response)
    
    # Format bold text (assuming it's used for sub-headers or important points)
    response = re.sub(r'\*\*(.*?)\*\*', r'<span style="color: #4CAF50; font-weight: bold;">\1</span>', response)
    
    return response

async def main():
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
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = DEFAULT_API_KEY
    if 'research_topic' not in st.session_state:
        st.session_state.research_topic = False
    if 'research_iterations' not in st.session_state:
        st.session_state.research_iterations = 2  # Default value

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
        progress_bar = st.sidebar.progress(0)
        iterations = st.session_state.train_iterations

        # Run training asynchronously
        training_task = asyncio.create_task(run_training(iterations, progress_bar))
        add_script_run_ctx(training_task)
        training_task.add_done_callback(handle_training_result)

    # Display training result
    if 'training_success' in st.session_state:
        if st.session_state.training_success:
            st.sidebar.success("Training completed successfully!")
        else:
            st.sidebar.error("Training failed.")
        # Clear the training_success flag
        del st.session_state.training_success

    # Rest of the main function (chat interface, etc.)
    col1, col2, col3 = st.sidebar.columns(3)

    if 'use_openai' not in st.session_state:
        st.session_state.use_openai = True
    if 'search_internet' not in st.session_state:
        st.session_state.search_internet = False

    def toggle_openai():
        logger.debug("Toggling OpenAI usage")
        st.session_state.use_openai = not st.session_state.use_openai
        if st.session_state.use_openai:
            st.session_state.search_internet = False
            st.session_state.research_topic = False

    def toggle_internet():
        logger.debug("Toggling Internet search")
        st.session_state.search_internet = not st.session_state.search_internet
        if st.session_state.search_internet:
            st.session_state.use_openai = False
            st.session_state.research_topic = False

    def toggle_research():
        logger.debug("Toggling research mode")
        st.session_state.research_topic = not st.session_state.research_topic
        if st.session_state.research_topic:
            st.session_state.use_openai = False
            st.session_state.search_internet = False

    use_openai = col1.checkbox("Use OpenAI", value=st.session_state.use_openai, on_change=toggle_openai)
    search_internet = col2.checkbox("Search Internet", value=st.session_state.search_internet, on_change=toggle_internet)
    research_topic_toggle = col3.checkbox("Research Topic", value=st.session_state.research_topic, on_change=toggle_research)

    # OpenAI API Key input
    if not st.session_state.get("openai_api_key_set"):
        openai_api_key = st.sidebar.text_input("OpenAI API Key (required)", type="password")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            set_openai_api_key(st.session_state.openai_api_key)
            logger.debug(f"Using OpenAI API key: {st.session_state.openai_api_key[:5]}...{st.session_state.openai_api_key[-5:]}")
            st.session_state.openai_api_key_set = True
        elif 'openai_api_key' not in st.session_state or not st.session_state.openai_api_key:
            st.session_state.openai_api_key = DEFAULT_API_KEY
            set_openai_api_key(st.session_state.openai_api_key)
            st.session_state.openai_api_key_set = True

    # Check if API key is provided
    if not st.session_state.openai_api_key:
        logger.error("OpenAI API key not provided")
        st.error("Please enter your OpenAI API key in the sidebar to use the OpenAI features.")
        return

    # Research Topic functionality
    if research_topic_toggle:
        research_iterations = st.sidebar.number_input(
            "Research Iterations",
            min_value=1,
            max_value=10,
            value=st.session_state.research_iterations,
            step=1,
            key="research_iterations_input"
        )
        st.session_state.research_iterations = research_iterations

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("What is your question?"):
        logger.debug(f"Received user prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    logger.debug(f"Processing user query: {prompt}")
                    logger.debug(f"Current OpenAI API key: {st.session_state.openai_api_key[:5]}...{st.session_state.openai_api_key[-5:]}")

                    if use_local_cache:
                        interactions = await load_interactions()
                        logger.debug(f"Loaded interactions: {interactions}")
                        similar_interaction, similarity_score = find_similar_interaction(prompt, interactions)
                        if similar_interaction:
                            response = "[Cached]\n" + similar_interaction["answer"]
                            logger.debug(f"Found similar interaction in cache: {response}")
                        else:
                            response = "[Cached]\nI don't have a cached answer for this query."
                            logger.debug("No cached answer found")
                    elif search_internet:
                        logger.debug("Searching the internet for the prompt")
                        scraped_data = await search_internet_and_process(prompt)
                        logger.debug(f"Internet search result: {scraped_data}")
                        response = "[Web]\n" + scraped_data['answer']
                        logger.debug(f"Generated response: {response}")
                    elif research_topic_toggle:
                        logger.debug(f"Researching topic: {prompt}")
                        researcher = Researcher()
                        research_results = await researcher.research_topic(prompt, st.session_state.research_iterations)

                        logger.debug(f"Type of research_results: {type(research_results)}, Content: {research_results}")

                        if research_results:
                            filtered_results = [
                                result for result in research_results
                                if not result['summary'].startswith("I'm sorry, but I couldn't find any")
                                and not result['summary'].startswith("I'm sorry, but I couldn't find any relevant information")
                            ]
                            
                            if filtered_results:
                                final_summary = await researcher.generate_final_summary(prompt, filtered_results)
                                response = "# Research Results\n\n"
                                for idx, result in enumerate(filtered_results, 1):
                                    response += f"## Research Result {idx}\n\n{result['summary']}\n\n"
                                response += f"# Final Summary\n\n{final_summary}"
                            else:
                                response = "No informative research results found."
                        else:
                            response = get_fallback_response()
                        logger.debug(f"Research results: {response}")
                    elif use_openai:
                        logger.debug("Attempting to get OpenAI response")
                        openai_response = await get_openai_response(prompt)
                        response = "[OpenAI]\n" + openai_response
                        logger.debug(f"OpenAI response: {response}")
                    else:
                        response = "Please enable either OpenAI, Internet Search, or Research Topic to get an answer."
                        logger.debug("No response generated: No option selected")

                    if filter_irrelevant_responses(response):
                        formatted_response = format_response(response)
                        st.markdown(formatted_response, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    else:
                        logger.debug("Filtered out an irrelevant response.")
                        st.warning("I couldn't find a relevant answer to your query. Please try rephrasing or asking a different question.")

                    # Save the interaction
                    await save_interaction(prompt, response)

                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"An error occurred while processing your query: {str(e)}")

    logger.debug(f"Final session state: {st.session_state}")
    logger.debug("Exiting main()")

if __name__ == "__main__":
    logger.debug("Starting the application")
    asyncio.run(main())