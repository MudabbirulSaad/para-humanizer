# Para-Humanizer API

A professional REST API for the Para-Humanizer paraphrasing service, designed with OpenAI-like conventions.

## Overview

The Para-Humanizer API provides HTTP endpoints that allow you to integrate the powerful paraphrasing capabilities of the Para-Humanizer tool into your applications. The API follows professional conventions similar to OpenAI's API design, making it familiar and easy to use for developers.

## Intelligent Parameter Selection

The API includes an intelligent parameter selection system that automatically determines the optimal paraphrasing parameters based on the characteristics of your input text. This feature analyzes various aspects of the text such as:

- Lexical diversity
- Sentence structure
- Formality level
- Technical content
- Text length
- Punctuation usage

To use intelligent parameter selection, simply use the default parameter values (or explicitly set the `--intelligent` flag in the example client). The API will automatically detect this and apply optimized parameters tailored to your specific text.

## Getting Started

### Installation

Ensure you have all the required dependencies:

```bash
pip install fastapi uvicorn requests
```

### Running the API Server

Start the API server using the provided script:

```bash
python run_api.py --host 0.0.0.0 --port 8000
```

Command-line options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development

### API Documentation

Once the server is running, you can access the auto-generated API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Security Features

The Para-Humanizer API includes several security features to help protect your deployment:

### API Key Authentication

You can enable API key authentication to restrict access to authorized users only. When enabled, clients must include a valid API key in the `X-API-Key` header with each request.

To enable API key authentication:

```bash
# Using command-line arguments
python run_api.py --api-keys "key1,key2,key3"

# Using environment variables
export PARA_HUMANIZER_API_KEYS="key1,key2,key3"
python run_api.py
```

### Rate Limiting

Rate limiting helps protect your API from abuse by limiting the number of requests a client can make in a specific time period. When enabled, clients exceeding the limit will receive a 429 Too Many Requests response.

To enable rate limiting:

```bash
# Using command-line arguments
python run_api.py --rate-limit 60  # 60 requests per minute

# Using environment variables
export PARA_HUMANIZER_RATE_LIMIT=60
python run_api.py
```

### CORS Configuration

The API supports Cross-Origin Resource Sharing (CORS) configuration to control which origins can access your API. By default, all origins are allowed, but you can restrict this for production use.

Configure CORS using the environment variable:

```bash
# Allow only specific origins
export PARA_HUMANIZER_CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
python run_api.py
```

## Configuration

The Para-Humanizer API can be configured through environment variables, command-line arguments, or a configuration file.

### Command-line Arguments

```bash
python run_api.py --help
```

Common options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--debug`: Enable debug mode
- `--config`: Path to a configuration file
- `--api-keys`: Comma-separated list of API keys
- `--rate-limit`: Rate limit in requests per minute

### Environment Variables

- `PARA_HUMANIZER_API_KEYS`: Comma-separated list of valid API keys
- `PARA_HUMANIZER_RATE_LIMIT`: Maximum requests per minute
- `PARA_HUMANIZER_CORS_ORIGINS`: Comma-separated list of allowed origins
- `PARA_HUMANIZER_DEBUG`: Set to "true" to enable debug mode
- `PARA_HUMANIZER_USE_GPU`: Set to "true" or "false" to control GPU usage

## API Endpoints

### Health Check

```
GET /
```

Returns the API status and version.

### Paraphrase Text

```
POST /v1/paraphrase
```

Paraphrases input text based on the provided parameters.

#### Request Body

```json
{
  "text": "Your text to paraphrase goes here.",
  "rule_based_rate": 0.4,
  "transformer_rate": 0.0,
  "humanize": true,
  "humanize_intensity": 0.5,
  "typo_rate": 0.0,
  "no_parallel": false
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | (required) | The input text to paraphrase |
| `rule_based_rate` | float | 0.4 | The rate of rule-based word replacement (0.0 to 1.0) |
| `transformer_rate` | float | 0.0 | The rate of transformer-based paraphrasing (0.0 to 1.0) |
| `humanize` | boolean | true | Whether to apply humanization techniques |
| `humanize_intensity` | float | 0.5 | The intensity of humanization (0.0 to 1.0) |
| `typo_rate` | float | 0.0 | The rate of introducing typos (0.0 to 1.0) |
| `no_parallel` | boolean | false | Whether to disable parallel processing |

#### Response

```json
{
  "id": "paraphrase-7aed8cef-c45e-4f4d-9fe3-5f4f7a3b8c2a",
  "object": "paraphrase",
  "created": 1710392400,
  "model": "ultimate-paraphraser-v1",
  "choices": [
    {
      "index": 0,
      "text": "Your paraphrased text result here.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 12,
    "total_tokens": 22
  }
}
```

## Example Usage

### Python Client

```python
import requests

url = "http://localhost:8000/v1/paraphrase"
payload = {
    "text": "The quick brown fox jumps over the lazy dog.",
    "rule_based_rate": 0.4,
    "transformer_rate": 0.0,
    "humanize": True,
    "humanize_intensity": 0.5,
    "typo_rate": 0.01
}

response = requests.post(url, json=payload)
result = response.json()
paraphrased_text = result["choices"][0]["text"]
print(paraphrased_text)
```

### Command-line Client

We provide an example client in `examples/api_client.py`:

```bash
python examples/api_client.py --file input.txt --rule-based-rate 0.4 --typo-rate 0.01
```

Command-line options:
- `--url`: API base URL (default: http://localhost:8000)
- `--text`: Text to paraphrase
- `--file`: File containing text to paraphrase
- `--rule-based-rate`: Rate of rule-based paraphrasing (0.0 to 1.0)
- `--transformer-rate`: Rate of transformer-based paraphrasing (0.0 to 1.0)
- `--no-humanize`: Disable humanization techniques
- `--humanize-intensity`: Intensity of humanization (0.0 to 1.0)
- `--typo-rate`: Rate of introducing typos (0.0 to 1.0)

## Error Handling

The API follows standard HTTP status codes for indicating success or failure. In case of an error, the response will have the following format:

```json
{
  "error": {
    "message": "Error message description",
    "type": "error_type",
    "param": null,
    "code": "error_code"
  }
}
```

## Deployment Considerations

For production deployment:

1. Add proper authentication
2. Configure CORS settings for your specific use case
3. Set up SSL/TLS for secure connections
4. Consider using a reverse proxy like Nginx
5. Implement rate limiting to prevent abuse
6. Add monitoring and logging for observability

## Advanced Usage

### Customizing the API

You can extend the API by modifying the `api/app.py` file to add new endpoints or enhance existing ones.

### Performance Tuning

Adjust batch sizes and parallelization settings based on your hardware capabilities:

```python
paraphraser = UltimateParaphraser(
    use_gpu=detect_gpu(),
    batch_size=4,  # Increase for more powerful systems
    enable_rule_based=True,
    enable_transformer=True
)

```

## Programmatic Usage

```python
from api.app import paraphraser

# Use the paraphraser directly
result = paraphraser.paraphrase(
    text="Your text to paraphrase",
    rule_based_rate=0.4,
    transformer_rate=0.0,
    humanize=True,
    humanize_intensity=0.5,
    typo_rate=0.0
)

print(result)
