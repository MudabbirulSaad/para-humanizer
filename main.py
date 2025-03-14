import nltk
from nltk.corpus import wordnet
import random
import spacy
import re
import os
import time
import argparse
import threading
import traceback
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from the new modular package structure
from para_humanizer.core.paraphraser import UltimateParaphraser


def main():
    parser = argparse.ArgumentParser(description='Advanced text paraphraser with human-like output')
    parser.add_argument('--input', '-i', type=str, help='Input file to paraphrase')
    parser.add_argument('--output', '-o', type=str, help='Output file for paraphrased text')
    parser.add_argument('--rule-based-rate', '-r', type=float, default=0.4, 
                        help='Rate of word replacement for rule-based approach (0.0 to 1.0)')
    parser.add_argument('--humanize-intensity', '-m', type=float, default=0.6,
                        help='Intensity of humanizing (0.0 to 1.0)')
    parser.add_argument('--typo-rate', '-t', type=float, default=0.01,
                        help='Rate of introducing realistic typos (0.0 to 1.0)')
    parser.add_argument('--no-humanize', action='store_true',
                        help='Disable humanizing effects')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-transformer', action='store_true',
                        help='Disable transformer model completely (Windows fix)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing (use sequential for better reliability)')
    parser.add_argument('--text', type=str,
                        help='Directly paraphrase provided text instead of using file')

    args = parser.parse_args()

    # Initialize the paraphraser from our new module
    paraphraser = UltimateParaphraser(
        use_gpu=not args.no_gpu,
        batch_size=2,  # Even lower batch size for better stability
        hybrid_mode=True,
        transformer_disable=args.no_transformer  # Force disable transformers on Windows
    )

    # Handle direct text input
    if args.text:
        print("Original text:")
        print(args.text)
        print("\nParaphrasing...")
        
        start_time = time.time()
        paraphrased = paraphraser.paraphrase(
            args.text,
            rule_based_rate=args.rule_based_rate,
            transformer_rate=0.0,  # Force 0 for Windows compatibility
            humanize=not args.no_humanize,
            humanize_intensity=args.humanize_intensity,
            typo_rate=args.typo_rate,
            no_parallel=args.no_parallel
        )
        end_time = time.time()
        
        print(f"Paraphrased in {end_time - start_time:.2f} seconds:")
        print(paraphrased)
        return

    # Handle file input/output
    if args.input:
        try:
            # Read the input file
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"Paraphrasing file: {args.input}")
            print(f"Text length: {len(text)} characters")
            
            # Process the text
            start_time = time.time()
            paraphrased = paraphraser.paraphrase(
                text,
                rule_based_rate=args.rule_based_rate,
                transformer_rate=0.0,  # Force 0 for Windows compatibility
                humanize=not args.no_humanize,
                humanize_intensity=args.humanize_intensity,
                typo_rate=args.typo_rate,
                no_parallel=args.no_parallel
            )
            end_time = time.time()
            
            # Determine output file
            output_file = args.output
            if not output_file:
                base, ext = os.path.splitext(args.input)
                output_file = f"{base}_paraphrased{ext}"
            
            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(paraphrased)
                
            print(f"Paraphrasing completed in {end_time - start_time:.2f} seconds")
            print(f"Output saved to: {output_file}")
            return True
        except Exception as e:
            import traceback
            print(f"Error paraphrasing file: {e}")
            print(traceback.format_exc())  # Print full stack trace
            return False

    # Run default examples if no input provided
    print("Running example paraphrasing...")
    
    # Test with simple examples
    original_text = "The quick brown fox jumps over the lazy dog. This sentence is used to demonstrate paraphrasing capabilities."
    
    print("Original:", original_text)
    
    # Default settings (balanced approach)
    print("\nDefault paraphrasing:")
    print(paraphraser.paraphrase(original_text, typo_rate=0.01))
    
    # More aggressive replacement
    print("\nMore aggressive paraphrasing:")
    print(paraphraser.paraphrase(
        original_text, 
        rule_based_rate=0.6, 
        transformer_rate=0.0,  # Force 0 for Windows compatibility
        humanize_intensity=0.7,
        typo_rate=0.02
    ))
    
    # Without humanizing
    print("\nWithout humanizing:")
    print(paraphraser.paraphrase(original_text, humanize=False))
    
    # Longer example
    longer_text = """
    Artificial intelligence is transforming how we work and live. 
    Machine learning algorithms can analyze vast amounts of data.
    They can identify patterns that humans might miss.
    This technology has applications in healthcare, finance, and transportation.
    However, there are concerns about privacy and job displacement.
    """
    
    print("\n--- Longer Example ---")
    print("Original:", longer_text)
    print("\nParaphrased:")
    print(paraphraser.paraphrase(longer_text))

if __name__ == "__main__":
    main()