## Requirements

### Hardware
- CUDA-capable GPU (recommended)
  - Minimum 8GB VRAM
  - 16GB+ VRAM recommended for optimal performance
- 16GB+ RAM recommended
- SSD for storage (minimum 20GB free space)

### Software
- Python 3.8+
- CUDA Toolkit 11.7+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full list

## Features

- 🧠 Long-term memory and context awareness
- 👤 Identity management system
- 🔍 Web search capabilities with caching
- 📈 Real-time stock data integration
- 💾 File handling with security measures
- 🚀 128K context window support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama3-chatqa-assistant.git
cd llama3-chatqa-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model:
```bash
python scripts/download_model.py
```

## Usage

### Basic Usage
```bash
python main.py
```

### CLI Interface
```bash
python cli.py --model nvidia/Llama3-ChatQA-2-8B
```

## Troubleshooting

### Common Issues
1. CUDA Out of Memory
   - Clear conversation history
   - Reduce context window size
   - Free GPU memory

2. Identity Not Updating
   - Check identities.json permissions
   - Verify memory persistence
   - Clear cache if needed

3. Slow Response Time
   - Check GPU utilization
   - Monitor memory usage
   - Clear conversation history

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## Acknowledgments

- NVIDIA for Llama3-ChatQA-2-8B
- Contributors and maintainers
- Open source community
