# AI Assistant

## Project Description

This project is an advanced Natural Language Processing (NLP) suite designed to query, summarize, and fact-check answers from the web using machine learning and API services. The suite includes various scripts to handle data analysis, database management, fact-checking, model training, NLP processing, and web scraping. The project aims to provide accurate and qualitative answers to user queries by leveraging free and open-source modules, GPU support with torch, and self-learning capabilities.

## Table of Contents

- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Requirements](#requirements)
- [Scripts and Usage](#scripts-and-usage)
  - [clear_cache.py](#clear_cachepy)
  - [data_analyzer.py](#data_analyzerpy)
  - [database_handler.py](#database_handlerpy)
  - [fact_checker.py](#fact_checkerpy)
  - [main.py](#mainpy)
  - [model_trainer.py](#model_trainerpy)
  - [nlp_processor.py](#nlp_processorpy)
  - [web_interface.py](#web_interfacepy)
  - [web_scraper.py](#web_scraperpy)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)
- [Additional Information](#additional-information)
- [Contact](#contact)

## Prerequisites

- Python 3.8+
- Conda
- PyTorch with GPU support
- Various Python libraries (see below)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/slyfox1186/ai-assistant.git
    cd ai-assistant
    ```

2. Create and activate a Conda environment:
    ```sh
    conda create --name ai_assistant python=3.12.2
    conda activate ai_assistant
    ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Requirements

Create a `requirements.txt` file with the following contents:

```plaintext
torch
transformers
fuzzywuzzy
python-Levenshtein
requests
beautifulsoup4
pandas
numpy
scikit-learn
flask
gunicorn
```

## Scripts and Usage

### clear_cache.py

**Flow:**
1. Import necessary modules: `os` and `shutil`.
2. Define the `clear_cache` function which:
   - Checks if the `__pycache__` directory exists.
   - If it exists, remove it using `shutil.rmtree`.
   - Print a message indicating whether the cache was cleared or if it did not exist.
3. If the script is run directly, call the `clear_cache` function.

**Usage:**
```sh
python clear_cache.py
```

**Expected Output:**
- If the cache directory exists, it will be deleted and a message "Cleared __pycache__" will be printed.
- If the cache directory does not exist, a message "__pycache__ does not exist" will be printed.

### data_analyzer.py

**Flow:**
1. Import the `pandas` library.
2. Define the `analyze_data` function which:
   - Takes a file path as an argument.
   - Reads the CSV file located at the provided file path into a pandas DataFrame.
   - Prints the first few rows of the data.
   - Prints a statistical summary of the data.
   - Handles file not found and other potential exceptions.
3. If the script is run directly, call the `analyze_data` function with a sample file path.

**Usage:**
```sh
python data_analyzer.py
```

**Expected Output:**
- The script will print the head (first few rows) of the dataset and a statistical summary including count, mean, std deviation, min, max, and percentiles.

### database_handler.py

**Flow:**
1. Import the `sqlite3` module.
2. Define the `create_connection` function which:
   - Takes a database file path as an argument.
   - Attempts to create a connection to the SQLite database.
   - Prints a message indicating the success or failure of the connection.
   - Returns the connection object if successful.
3. Define the `create_table` function which:
   - Takes a connection object as an argument.
   - Executes a SQL command to create a table if it does not already exist.
   - Prints a message indicating the success or failure of the table creation.
4. If the script is run directly:
   - Create a connection to a sample database file.
   - If the connection is successful, create a table in the database.
   - Close the database connection.

**Usage:**
```sh
python database_handler.py
```

**Expected Output:**
- The script will print messages indicating whether the database connection was established and whether the table was created.

### fact_checker.py

**Flow:**
1. Import the `logging` module and the `DatabaseHandler` class from `database_handler`.
2. Set up logging configuration and create a logger.
3. Define the `FactChecker` class which:
   - Takes a `DatabaseHandler` object as an argument during initialization.
   - Contains the `check` method which:
     - Logs the start of the fact-checking process.
     - Iterates over provided data items.
     - Checks if the data item contains a fact (placeholder logic).
     - Logs a warning for potential misinformation.
     - Saves the number of checked facts to the database.
     - Returns the checked data.
4. If the script is run directly:
   - Create a `DatabaseHandler` object with a sample database file.
   - Create a `FactChecker` object with the database handler.
   - Define sample data and run the `check` method.
   - Log the fact-checked data and close the database connection.

**Usage:**
```sh
python fact_checker.py
```

**Expected Output:**
- The script will log messages indicating the start of the fact-checking process, any potential misinformation, and the number of checked facts.

### main.py

**Flow:**
1. Import necessary modules and functions: `argparse`, `analyze_data` from `data_analyzer`, `create_connection` and `create_table` from `database_handler`, and `fact_check` from `fact_checker`.
2. Define the `main` function which:
   - Takes a query as an argument.
   - Logs the query being processed.
   - Analyzes sample data using the `analyze_data` function.
   - Creates a connection to a sample database file.
   - If the connection is successful, creates a table in the database and closes the connection.
   - Fact-checks the query using the `fact_check` function.
3. If the script is run directly:
   - Parse command-line arguments to get the query.
   - Call the `main` function with the parsed query.

**Usage:**
```sh
python main.py --query "Your question here"
```

**Expected Output:**
- The script will log the query being processed, the result of data analysis, database operations, and the fact-checking outcome.

### model_trainer.py

**Flow:**
1. Import necessary components from the `transformers` library: `Trainer` and `TrainingArguments`.
2. Define the `train_model` function which:
   - Takes a data file path as an argument.
   - Sets up training arguments (e.g., output directory, number of epochs, batch size).
   - Initializes a `Trainer` object with the training arguments and dataset.
   - Starts the training process.
3. If the script is run directly, call the `train_model` function with a sample data file path.

**Usage:**
```sh
python model_trainer.py
```

**Expected Output:**
- The script will print training progress and results, including any relevant metrics.

### nlp_processor.py

**Flow:**
1. Import the `pipeline` function from the `transformers` library.
2. Define the `process_nlp` function which:
   - Takes a query as an argument.
   - Initializes a pre-trained question-answering pipeline.
   - Processes the query with a sample context using the pipeline.
   - Prints the answer returned by the pipeline.
3. If the script is run directly, call the `process_nlp` function with a sample query.

**Usage:**
```sh
python nlp_processor.py
```

**Expected Output:**
- The script will print the answer to the provided query based on the pre-trained model.

### web_interface.py

**Flow:**
1. Import necessary components from the `flask` library: `Flask`, `request`, and `jsonify`.
2. Import the `main` function from the `main` module.
3. Initialize a Flask application.
4. Define a route for POST requests to `/query` which:
   - Gets the query from the JSON data in the request.
   - If a query is provided, processes it using the

 `main` function and returns the result as JSON.
   - If no query is provided, returns an error message as JSON.
5. If the script is run directly, start the Flask application in debug mode.

**Usage:**
```sh
python web_interface.py
```

**Expected Output:**
- The Flask app will run and accept POST requests to `/query`, returning the processed result or an error message in JSON format.

### web_scraper.py

**Flow:**
1. Import necessary modules: `requests` and `BeautifulSoup` from `bs4`.
2. Define the `scrape_web` function which:
   - Takes a query as an argument.
   - Constructs a search URL using the query.
   - Sends a GET request to the search URL.
   - Parses the HTML content of the response using BeautifulSoup.
   - Prints the prettified HTML content.
3. If the script is run directly, call the `scrape_web` function with a sample query.

**Usage:**
```sh
python web_scraper.py
```

**Expected Output:**
- The script will print the prettified HTML content of the web page corresponding to the search query.

## Contribution Guidelines

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Additional Information

- **Features**: This suite includes self-learning capabilities, GPU support, and multi-source data fetching.
- **Future Plans**: Continuous improvement of the NLP models and integration with more data sources.
- **Acknowledgments**: Thanks to the open-source community for providing the necessary tools and libraries.

## Contact

For any questions or support, please contact [slyfox1186](https://github.com/slyfox1186).
