This will start the Streamlit server and open the AI Assistant interface in your default web browser.

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

We hope you find the AI Assistant useful for your NLP tasks. Your feedback and contributions are highly appreciated!
