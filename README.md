# AI Assistant with Persistent Memory

This AI assistant uses Redis Stack for persistent memory storage, vector similarity search, and JSON document storage. It includes RedisInsight for visual management and monitoring.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git
- At least 4GB RAM available
- Internet connection for first-time model download
- Hugging Face account (optional, for faster downloads)

## Initial Setup

1. **Install Required Python Packages**
   ```bash
   pip install -r config/requirements.txt
   pip install -U sentence-transformers huggingface-hub[cli]
   ```

2. **Download Sentence Transformer Model**
   ```bash
   # Optional: Login to Hugging Face for faster downloads
   huggingface-cli login
   
   # Download and cache the model (requires internet connection first time)
   python -c '
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
   print(f"Model downloaded and cached. Vector size: {len(model.encode('Test'))}")
   '
   ```
   The model (approximately 1.2GB) will be cached in:
   - Linux/Mac: `~/.cache/torch/sentence_transformers/`
   - Windows: `C:\Users\<username>\.cache\torch\sentence_transformers\`

   Note: The first run requires an internet connection to download the model. Subsequent runs will use the cached model.

3. **Install Docker and Docker Compose**
   ```