"""
Command-line interface for UltimateParaphraser.
Provides a CLI for using the paraphraser from the terminal.
"""
import argparse
import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

from para_humanizer.core.paraphraser import UltimateParaphraser
from para_humanizer.utils.config import DEFAULT_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="UltimateParaphraser: Advanced text paraphrasing with human-like qualities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_exclusive = input_group.add_mutually_exclusive_group(required=True)
    input_exclusive.add_argument(
        "-t", "--text", 
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
        help="Path to output file (if not specified, output to console)"
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
        default=DEFAULT_SETTINGS["batch_size"],
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
        "--rule-rate", 
        type=float,
        default=DEFAULT_SETTINGS["rule_based_rate"],
        help="Intensity of rule-based modifications (0.0 to 1.0)"
    )
    para_group.add_argument(
        "--transformer-rate", 
        type=float,
        default=DEFAULT_SETTINGS["transformer_rate"],
        help="Rate of transformer-based paraphrasing (0.0 to 1.0)"
    )
    para_group.add_argument(
        "--no-humanize", 
        action="store_true",
        help="Disable humanization effects"
    )
    para_group.add_argument(
        "--humanize-intensity", 
        type=float,
        default=DEFAULT_SETTINGS["humanize_intensity"],
        help="Intensity of humanization effects (0.0 to 1.0)"
    )
    para_group.add_argument(
        "--typo-rate", 
        type=float,
        default=DEFAULT_SETTINGS["typo_rate"],
        help="Rate of introducing realistic typos (0.0 to 1.0)"
    )
    
    # Parse the arguments
    return parser.parse_args()


def load_input_text(args: argparse.Namespace) -> str:
    """
    Load text from input file or command line.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Text to paraphrase
    """
    if args.text:
        return args.text
    elif args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            sys.exit(1)
    else:
        logger.error("No input provided")
        sys.exit(1)


def save_output_text(text: str, args: argparse.Namespace, original_text: Optional[str] = None) -> None:
    """
    Save or display the paraphrased text.
    
    Args:
        text: Paraphrased text
        args: Parsed command-line arguments
        original_text: Original text (for comparison)
    """
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.compare and original_text:
                    f.write("=== ORIGINAL TEXT ===\n\n")
                    f.write(original_text)
                    f.write("\n\n=== PARAPHRASED TEXT ===\n\n")
                f.write(text)
            logger.info(f"Output saved to {args.output}")
        except Exception as e:
            logger.error(f"Error writing to output file: {str(e)}")
            sys.exit(1)
    else:
        # Display to console
        if args.compare and original_text:
            print("=== ORIGINAL TEXT ===\n")
            print(original_text)
            print("\n=== PARAPHRASED TEXT ===\n")
        print(text)


def get_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Extract paraphraser settings from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of settings for the paraphraser
    """
    return {
        "use_gpu": not args.no_gpu,
        "batch_size": args.batch_size,
        "hybrid_mode": not args.no_hybrid,
        "transformer_disable": args.disable_transformer,
        "rule_based_rate": args.rule_rate,
        "transformer_rate": args.transformer_rate,
        "humanize": not args.no_humanize,
        "humanize_intensity": args.humanize_intensity,
        "typo_rate": args.typo_rate,
        "no_parallel": args.no_parallel
    }


def main() -> None:
    """Main function for the CLI."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load input text
    original_text = load_input_text(args)
    
    # Extract settings
    settings = get_settings_from_args(args)
    
    # Display information
    logger.info("UltimateParaphraser CLI")
    logger.info(f"Text size: {len(original_text)} characters")
    
    # Initialize paraphraser
    start_time = time.time()
    paraphraser = UltimateParaphraser(**settings)
    
    # Process the text
    logger.info("Processing text...")
    paraphrased_text = paraphraser.paraphrase(original_text)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    logger.info(f"Finished in {processing_time:.2f} seconds")
    
    # Save or display the result
    save_output_text(paraphrased_text, args, original_text if args.compare else None)


if __name__ == "__main__":
    main()
