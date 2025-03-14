#!/usr/bin/env python
"""
Example of processing text files with UltimateParaphraser.
This demonstrates how to paraphrase text from a file and save the results.
"""
import os
import time
from para_humanizer.core.paraphraser import UltimateParaphraser


def main():
    # Create a sample input file
    input_file = "examples/sample_input.txt"
    output_file = "examples/sample_output.txt"
    
    # Create sample content if input file doesn't exist
    if not os.path.exists(input_file):
        sample_content = """
        Natural language processing (NLP) is a field of artificial intelligence 
        that focuses on the interaction between computers and humans through natural language.
        
        The ultimate goal of NLP is to enable computers to understand, interpret, and generate
        human language in a way that is valuable. NLP combines computational linguistics, machine
        learning, and deep learning models to process human language.
        
        Applications of NLP include:
        - Machine translation
        - Sentiment analysis
        - Chatbots and virtual assistants
        - Text summarization
        - Speech recognition
        
        As technology advances, NLP systems become increasingly sophisticated in their ability
        to understand context, detect nuances, and generate human-like text responses.
        """
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        
        # Write sample content to file
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"Created sample input file: {input_file}")
    
    # Initialize the paraphraser
    print("Initializing UltimateParaphraser...")
    paraphraser = UltimateParaphraser(
        use_gpu=True,
        transformer_disable=True  # Disable transformer for simplified example
    )
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\nProcessing file: {input_file}")
    print(f"Input text length: {len(text)} characters")
    
    # Paraphrase the text with structure preservation
    start_time = time.time()
    paraphrased = paraphraser.paraphrase(
        text,
        rule_based_rate=0.5,
        humanize=True,
        humanize_intensity=0.6,
        typo_rate=0.01,
        preserve_structure=True  # Enable structure preservation to maintain formatting
    )
    end_time = time.time()
    
    # Write output to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== ORIGINAL TEXT ===\n\n")
        f.write(text)
        f.write("\n\n=== PARAPHRASED TEXT (WITH STRUCTURE PRESERVATION) ===\n\n")
        f.write(paraphrased)
        
        # Also add an example without structure preservation
        f.write("\n\n=== PARAPHRASED TEXT (WITHOUT STRUCTURE PRESERVATION) ===\n\n")
        no_structure_paraphrased = paraphraser.paraphrase(
            text,
            rule_based_rate=0.5,
            humanize=True,
            humanize_intensity=0.6,
            typo_rate=0.01,
            preserve_structure=False  # Disable structure preservation
        )
        f.write(no_structure_paraphrased)
    
    print(f"Paraphrasing completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_file}")
    print("\nFirst 200 characters of paraphrased text (with structure preservation):")
    print(paraphrased[:200] + "...")


if __name__ == "__main__":
    main()
