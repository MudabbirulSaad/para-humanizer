# UltimateParaphraser

Advanced text paraphrasing with human-like qualities.

## Overview

UltimateParaphraser is a powerful text transformation tool that paraphrases content while preserving its meaning and making it more human-like. It combines rule-based methods with optional transformer models to produce high-quality paraphrasing with natural human writing patterns.

### Key Features

- **GPU-accelerated processing** for large documents
- **Advanced humanization techniques** to mimic human writing patterns
- **Fine-grained control** over paraphrasing intensity and style
- **Self-hosted solution** without subscription fees
- **Modular architecture** for maintainability and extensibility
- **Multiple Paraphrasing Methods**: Choose between rule-based and transformer-based approaches, or use both in a hybrid mode.
- **Humanization Features**: Add realistic human-like variations to text.
- **Customizable Synonym Library**: Easily add, edit and manage synonyms using the built-in `SynonymManager`.
- **Self-Learning Synonyms**: Automatically discovers and learns new synonyms based on context and usage patterns.
- **GPU Acceleration**: Utilizes available GPU for faster processing of transformer models.
- **Batch Processing**: Efficiently process large documents with automatic chunking.
- **Language Preservation**: Maintains technical terminology and proper nouns.
- **API Integration**: Access paraphrasing capabilities remotely through a professional REST API.

## Installation

### Prerequisites

- Python 3.7 or higher
- For GPU acceleration: CUDA-compatible GPU with appropriate drivers

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/para-humanizer.git
   cd para-humanizer
   ```

2. Install the package:
   ```
   pip install -e .
   ```

3. Download required NLTK data and spaCy model:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   
   import spacy
   spacy.cli.download("en_core_web_sm")
   ```

## Usage

### Command Line Interface

The package provides a command-line interface for easy use:

```bash
# Paraphrase text from a file
python run_paraphraser.py -i input.txt -o output.txt

# Paraphrase direct text input
python run_paraphraser.py -t "Text to paraphrase"

# Adjust rule-based paraphrasing intensity
python run_paraphraser.py -i input.txt --rule-rate 0.6

# Control humanization intensity
python run_paraphraser.py -i input.txt --humanize-intensity 0.7

# Adjust typo frequency for more human-like output
python run_paraphraser.py -i input.txt --typo-rate 0.01

# Disable humanization
python run_paraphraser.py -i input.txt --no-humanize

# Force CPU-only mode
python run_paraphraser.py -i input.txt --no-gpu

# Disable transformer models (Windows compatibility)
python run_paraphraser.py -i input.txt --disable-transformer
```

### Python API

You can also use UltimateParaphraser in your Python code:

```python
from para_humanizer.core.paraphraser import UltimateParaphraser

# Initialize the paraphraser
paraphraser = UltimateParaphraser(
    use_gpu=True,               # Use GPU if available
    batch_size=8,               # Batch size for processing
    hybrid_mode=True,           # Use both rule-based and transformer approaches
    transformer_disable=False   # Enable transformer models
)

# Paraphrase text
text = "This is an example text to be paraphrased."
paraphrased = paraphraser.paraphrase(
    text,
    rule_based_rate=0.4,        # Rate of word replacement (0.0 to 1.0)
    transformer_rate=0.0,       # Rate of transformer usage (0.0 to 1.0)
    humanize=True,              # Apply humanization
    humanize_intensity=0.5,     # Intensity of humanization (0.0 to 1.0)
    typo_rate=0.005             # Rate of typo introduction (0.0 to 1.0)
)

print(paraphrased)

# Only apply humanization to text without paraphrasing
humanized = paraphraser.humanize(text, intensity=0.5)
print(humanized)
```

## Using the Synonym System

### Basic Usage with Synonyms

The `UltimateParaphraser` automatically uses the synonym library during paraphrasing:

```python
from para_humanizer import UltimateParaphraser

paraphraser = UltimateParaphraser()
text = "The algorithm efficiently processes data and generates accurate results."
paraphrased = paraphraser.paraphrase(text, rule_based_rate=0.4)
print(paraphrased)
# Output might be: "The algorithm effectively handles data and produces precise results."
```

### Managing Synonyms

Use the `SynonymManager` utility to manage your synonym library:

```python
from para_humanizer.utils.synonym_manager import SynonymManager

# Initialize the manager
manager = SynonymManager()

# Add new synonyms
manager.add_synonym("algorithm", "procedure", category="technical")
manager.add_synonym("generates", "produces", category="verbs")

# Remove synonyms
manager.remove_synonym("algorithm", "procedure")

# List all synonyms for a word
synonyms = manager.list_synonyms("algorithm")
print(synonyms)

# Export and import synonyms
manager.export_synonyms("my_synonyms.json")
manager.import_synonyms("my_synonyms.json")
```

### Self-Learning Synonym System

UltimateParaphraser now includes a self-learning capability that automatically discovers and adds new synonyms based on context and usage patterns:

```python
from para_humanizer import UltimateParaphraser

# Enable self-learning (enabled by default)
paraphraser = UltimateParaphraser(enable_learning=True)

# As you use the paraphraser, it will learn new synonyms automatically
text = "The engineers designed an efficient algorithm to handle complex calculations."
paraphrased = paraphraser.paraphrase(text)

# Get statistics about the learning process
stats = paraphraser.get_learning_stats()
print(stats)

# Reset learning statistics if needed
paraphraser.reset_learning_stats()
```

#### How Self-Learning Works

The self-learning system:

1. **Analyzes Context**: Examines words in their context to understand relationships
2. **Identifies Potential Synonyms**: Uses word embeddings and natural language techniques to find potential synonyms
3. **Validates Candidates**: Validates candidates against existing usage patterns
4. **Reinforces Through Usage**: Strengthens synonym connections based on successful usage
5. **Filters Results**: Automatically filters against blacklisted and common words

The system learns incrementally, improving its synonym library over time as you use the paraphraser.

## API Integration

Para-Humanizer now includes a professional REST API that allows you to integrate the paraphrasing capabilities into your own applications or services.

### Features

- **OpenAI-like API**: Familiar design pattern for easy integration
- **Intelligent Parameter Selection**: Automatically determines optimal paraphrasing parameters based on your text
- **Security Features**: API key authentication, rate limiting, and CORS configuration
- **Configurable Deployment**: Flexible configuration options for different environments

### Running the API Server

Start the API server with default settings:

```bash
python run_api.py
```

Or customize the configuration:

```bash
python run_api.py --host 0.0.0.0 --port 8000 --api-keys "your-api-key-here"
```

### API Documentation

For detailed information about the API endpoints, request/response formats, and configuration options, see the [API_README.md](./API_README.md).

### Example API Client

A Python client example is included to demonstrate how to interact with the API:

```bash
# Basic usage with default parameters
python examples/api_client.py --url http://localhost:8000 --file input.txt

# Using intelligent parameter selection
python examples/api_client.py --url http://localhost:8000 --file input.txt --intelligent
```

## Package Structure

The codebase follows a modular architecture for better maintainability:

```
para_humanizer/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── paraphraser.py         # Main paraphraser class
├── processors/
│   ├── __init__.py
│   ├── rule_based.py          # Rule-based text transformations
│   ├── humanizer.py           # Human-like text enhancements
│   └── transformer.py         # AI model-based paraphrasing
├── utils/
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── text_utils.py          # Text processing utilities
│   └── hardware.py            # Hardware detection and optimization
└── cli/
    ├── __init__.py
    └── command_line.py        # Command-line interface
```

## Advanced Configuration

### Customizing Protected Terms

You can add domain-specific terms to the protected list in `para_humanizer/utils/config.py` to prevent them from being modified during paraphrasing.

### JSON-based Synonym Library

UltimateParaphraser now uses a dynamic JSON-based synonym library located at `para_humanizer/resources/synonyms.json`. This approach offers several advantages:

- **Easy to maintain**: Update synonyms without modifying code
- **Categorized structure**: Synonyms are organized by parts of speech and categories
- **Expandable**: Add new categories and terms as needed
- **Filtered quality**: Automatically filters inappropriate terms using blacklist

To extend the synonym library:

1. Directly edit the JSON file following the existing structure
2. Use the synonym management utility:

```bash
# Add a new synonym
python -m para_humanizer.utils.synonym_manager add --word "excellent" --synonym "outstanding" --category "adjectives"

# Remove a synonym
python -m para_humanizer.utils.synonym_manager remove --word "excellent" --synonym "outstanding"

# Export the current library
python -m para_humanizer.utils.synonym_manager export --output custom_synonyms.json
```

### Adjusting Synonym Quality

The synonym library works with the blacklist and common words filters in `para_humanizer/utils/config.py`. You can adjust these lists to fine-tune which words get replaced during paraphrasing.

### Windows Compatibility

For Windows systems, transformer models might cause issues with CUDA. The default configuration sets `transformer_rate=0.0` for Windows compatibility, but you can experiment with enabling it based on your specific environment.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce the batch size with `--batch-size 2` or lower.
2. **Slow Processing**: Enable parallel processing by ensuring `--no-parallel` is not set.
3. **Installation Problems**: Ensure you have the correct versions of PyTorch and transformers for your CUDA setup.

## License

MIT License

Copyright (c) 2025 Para-Humanizer Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- NLTK for natural language processing capabilities
- Hugging Face Transformers for pre-trained language models
- SpaCy for advanced text processing
