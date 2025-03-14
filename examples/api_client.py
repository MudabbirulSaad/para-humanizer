"""
Example client for the Para-Humanizer API.
Demonstrates how to make requests to the API endpoint.
"""
import requests
import json
import argparse
import sys


def paraphrase_text(api_url, text, rule_based_rate=0.4, transformer_rate=0.0, 
                    humanize=True, humanize_intensity=0.5, typo_rate=0.0,
                    use_intelligent_params=False, preserve_structure=True):
    """
    Send a paraphrase request to the API.
    
    Args:
        api_url: Base URL of the Para-Humanizer API
        text: Text to paraphrase
        rule_based_rate: Rate of rule-based paraphrasing (0.0 to 1.0)
        transformer_rate: Rate of transformer-based paraphrasing (0.0 to 1.0)
        humanize: Whether to apply humanization techniques
        humanize_intensity: Intensity of humanization (0.0 to 1.0)
        typo_rate: Rate of introducing typos (0.0 to 1.0)
        use_intelligent_params: Whether to use intelligent parameter selection
        preserve_structure: Whether to preserve the original document structure
        
    Returns:
        Paraphrased text if successful, None otherwise
    """
    # Build the request URL
    url = f"{api_url}/v1/paraphrase"
    
    # If using intelligent parameters, set all parameters to their default values
    # The API will detect this and apply intelligent parameter selection
    if use_intelligent_params:
        rule_based_rate = 0.4
        transformer_rate = 0.0
        humanize_intensity = 0.5
        typo_rate = 0.0
    
    # Prepare the request payload
    payload = {
        "text": text,
        "rule_based_rate": rule_based_rate,
        "transformer_rate": transformer_rate,
        "humanize": humanize,
        "humanize_intensity": humanize_intensity,
        "typo_rate": typo_rate,
        "no_parallel": False,
        "preserve_structure": preserve_structure
    }
    
    try:
        # Send the request
        response = requests.post(
            url, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["text"]
        else:
            error = response.json().get("error", {"message": "Unknown error"})
            print(f"Error: {error.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse API response")
        return None


def main():
    """Process command line arguments and make API request."""
    parser = argparse.ArgumentParser(description="Para-Humanizer API Client")
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8000", 
        help="API base URL"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        help="Text to paraphrase"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="File containing text to paraphrase"
    )
    parser.add_argument(
        "--intelligent", 
        action="store_true",
        help="Use intelligent parameter selection based on text analysis"
    )
    parser.add_argument(
        "--rule-based-rate", 
        type=float, 
        default=0.4,
        help="Rate of rule-based paraphrasing (0.0 to 1.0)"
    )
    parser.add_argument(
        "--transformer-rate", 
        type=float, 
        default=0.0,
        help="Rate of transformer-based paraphrasing (0.0 to 1.0)"
    )
    parser.add_argument(
        "--no-humanize", 
        action="store_false",
        dest="humanize",
        help="Disable humanization techniques"
    )
    parser.add_argument(
        "--humanize-intensity", 
        type=float, 
        default=0.5,
        help="Intensity of humanization (0.0 to 1.0)"
    )
    parser.add_argument(
        "--typo-rate", 
        type=float, 
        default=0.0,
        help="Rate of introducing typos (0.0 to 1.0)"
    )
    parser.add_argument(
        "--preserve-structure", 
        action="store_true",
        help="Preserve the original document structure"
    )
    
    args = parser.parse_args()
    
    # Get input text either from command line or file
    if args.text:
        input_text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Please provide text either via --text or --file")
        return
    
    # Call the API
    if args.intelligent:
        print("Using intelligent parameter selection based on text analysis...")
        result = paraphrase_text(
            api_url=args.url,
            text=input_text,
            use_intelligent_params=True
        )
    else:
        result = paraphrase_text(
            api_url=args.url,
            text=input_text,
            rule_based_rate=args.rule_based_rate,
            transformer_rate=args.transformer_rate,
            humanize=args.humanize,
            humanize_intensity=args.humanize_intensity,
            typo_rate=args.typo_rate,
            use_intelligent_params=False,
            preserve_structure=args.preserve_structure
        )
    
    # Print the result
    if result:
        print("\nParaphrased text:")
        print("----------------")
        print(result)
        print("----------------")


if __name__ == "__main__":
    main()
