#!/usr/bin/env python
"""
UltimateParaphraser - Advanced text paraphrasing with human-like qualities.
Entry point script for the refactored module structure.
"""
import argparse
import os
import sys
import time

from para_humanizer.core.paraphraser import UltimateParaphraser


def main():
    """Main function that handles command-line operation."""
    parser = argparse.ArgumentParser(
        description='UltimateParaphraser: Advanced text paraphrasing with human-like output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_exclusive = input_group.add_mutually_exclusive_group(required=False)
    input_exclusive.add_argument(
        "--text", 
        type=str,
        help="Direct text input to paraphrase"
    )
    input_exclusive.add_argument(
        "-i", "--input", 
        type=str,
        help="Path to input file containing text to paraphrase"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output", 
        type=str,
        default=None,
        help="Path to output file (if not specified, output to console or add '_paraphrased' to input file name)"
    )
    output_group.add_argument(
        "--compare", 
        action="store_true",
        help="Show original and paraphrased text side by side"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--no-gpu", 
        action="store_true",
        help="Disable GPU usage even if available"
    )
    proc_group.add_argument(
        "--batch-size", 
        type=int,
        default=2,
        help="Batch size for processing"
    )
    proc_group.add_argument(
        "--no-hybrid", 
        action="store_true",
        help="Disable hybrid mode (use only rule-based methods)"
    )
    proc_group.add_argument(
        "--disable-transformer", 
        action="store_true",
        help="Disable transformer models completely"
    )
    proc_group.add_argument(
        "--no-parallel", 
        action="store_true",
        help="Disable parallel processing"
    )
    
    # Paraphrasing options
    para_group = parser.add_argument_group("Paraphrasing Options")
    para_group.add_argument(
        "--rule-rate", "-r", 
        type=float,
        default=0.4,
        help="Intensity of rule-based modifications (0.0 to 1.0)"
    )
    para_group.add_argument(
        "--transformer-rate", 
        type=float,
        default=0.0,  # Default to 0 for Windows compatibility
        help="Rate of transformer-based paraphrasing (0.0 to 1.0)"
    )
    para_group.add_argument(
        "--no-humanize", 
        action="store_true",
        help="Disable humanization effects"
    )
    para_group.add_argument(
        "--humanize-intensity", "-m", 
        type=float,
        default=0.5,
        help="Intensity of humanization effects (0.0 to 1.0)"
    )
    para_group.add_argument(
        "--typo-rate", "-y", 
        type=float,
        default=0.005,
        help="Rate of introducing realistic typos (0.0 to 1.0)"
    )

    args = parser.parse_args()

    # Initialize the paraphraser
    paraphraser = UltimateParaphraser(
        use_gpu=not args.no_gpu,
        batch_size=args.batch_size,
        hybrid_mode=not args.no_hybrid,
        transformer_disable=args.disable_transformer
    )

    # Handle direct text input
    if args.text:
        print("Original text:")
        print(args.text)
        print("\nParaphrasing...")
        
        start_time = time.time()
        paraphrased = paraphraser.paraphrase(
            args.text,
            rule_based_rate=args.rule_rate,
            transformer_rate=args.transformer_rate,
            humanize=not args.no_humanize,
            humanize_intensity=args.humanize_intensity,
            typo_rate=args.typo_rate,
            no_parallel=args.no_parallel
        )
        end_time = time.time()
        
        print(f"Paraphrased in {end_time - start_time:.2f} seconds:")
        if args.compare:
            print("\n=== ORIGINAL TEXT ===\n")
            print(args.text)
            print("\n=== PARAPHRASED TEXT ===\n")
        print(paraphrased)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(paraphrased)
            print(f"\nOutput saved to: {args.output}")
        
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
                rule_based_rate=args.rule_rate,
                transformer_rate=args.transformer_rate,
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
                if args.compare:
                    f.write("=== ORIGINAL TEXT ===\n\n")
                    f.write(text)
                    f.write("\n\n=== PARAPHRASED TEXT ===\n\n")
                f.write(paraphrased)
                
            print(f"Paraphrasing completed in {end_time - start_time:.2f} seconds")
            print(f"Output saved to: {output_file}")
            return
            
        except Exception as e:
            import traceback
            print(f"Error paraphrasing file: {e}")
            print(traceback.format_exc())
            return

    # Run demo examples if no input or text provided
    run_demo_examples(paraphraser)


def run_demo_examples(paraphraser):
    """Run demonstration examples with the paraphraser."""
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
