# AI Assistant: Advanced NLP Suite with Streamlit GUI

## Project Description

AI Assistant is an advanced Natural Language Processing (NLP) suite designed to provide intelligent, context-aware responses to user queries. This project leverages cutting-edge machine learning techniques, web scraping, and the OpenAI API to deliver accurate and comprehensive answers. The system features a user-friendly Streamlit GUI, making it accessible for both technical and non-technical users.

## Key Features

- **Streamlit GUI**: Intuitive web interface for easy interaction
- **OpenAI Integration**: Utilizes GPT models for advanced language understanding and generation
- **Web Scraping**: Fetches and processes information from various online sources
- **Local Caching**: Stores and retrieves previous interactions for faster responses
- **Self-Learning Capability**: Improves performance over time through continuous learning
- **GPU Acceleration**: Harnesses CUDA-enabled GPUs for enhanced processing speed
- **Fact-Checking**: Implements robust fact-checking mechanisms for information accuracy
- **Customizable**: Easily adaptable for various NLP tasks and domains

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Downloading Required Models](#downloading-required-models)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/slyfox1186/ai-assistant.git
   cd ai-assistant
   ```

2. Create and activate a Conda environment:
   ```
   conda create --name ai_assistant python=3.12.2
   conda activate ai_assistant
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `config.py` file in the project root
   - Add your API key: `OPENAI_API_KEY = "your-api-key-here"`

## Downloading Required Models

Before running the AI Assistant, you need to download the necessary NLP models. Run the following commands:

1. Download the spaCy model:
   ```
   python -m spacy download en_core_web_trf
   ```

2. Download the Flair model:
   ```
   python -m flair.models.text_classification.sentiment download
   ```

3. Download the Hugging Face transformers models:
   ```
   python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
   ```

These commands will download and cache the required models for NER processing, sentiment analysis, and similarity checking.

## Usage

To launch the AI Assistant with the Streamlit GUI, run:

```
streamlit run main.py
```

This will start the Streamlit server and open the AI Assistant interface in your default web browser. By default, it will run on `http://localhost:8501`.

If you want to specify a different port or host, you can use:

```
streamlit run main.py --server.port 8080 --server.address 0.0.0.0
```

### Using the Interface

1. **API Key**: Enter your OpenAI API key in the sidebar (if not set in `config.py`)
2. **Query Input**: Type your question in the chat input at the bottom of the page
3. **Response**: The AI Assistant will process your query and display the answer
4. **Settings**: Use the sidebar to customize behavior:
   - Toggle between using OpenAI API or internet search
   - Enable/disable local cache usage
   - Adjust model training parameters

## Scripts Overview

- `main.py`: The core script that launches the Streamlit interface and orchestrates the entire process
- `openai_handler.py`: Manages interactions with the OpenAI API
- `web_scraper.py`: Handles web scraping for additional information
- `data_manager.py`: Manages data storage and retrieval, including local caching
- `similarity_checker.py`: Compares queries to find similar previous interactions
- `model_trainer.py`: Handles the training and updating of local models
- `ner_processor.py`: Processes named entities in the text

## Configuration

Adjust the following files to customize the AI Assistant:

- `config.py`: Set API keys, model parameters, and other global settings
- `requirements.txt`: Manage project dependencies

## Contributing

We welcome contributions to the AI Assistant project! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Project Link: [https://github.com/slyfox1186/ai-assistant](https://github.com/slyfox1186/ai-assistant)

For support or queries, please open an issue in the GitHub repository or contact [slyfox1186](https://github.com/slyfox1186).

---

We hope you find the AI Assistant useful for your NLP tasks. Your feedback and contributions are highly app
