#!/usr/bin/env python
"""
Basic example of using the UltimateParaphraser.
This demonstrates simple text paraphrasing with different settings.
"""
from para_humanizer.core.paraphraser import UltimateParaphraser


def main():
    # Initialize the paraphraser
    print("Initializing UltimateParaphraser...")
    paraphraser = UltimateParaphraser(
        use_gpu=True,  # Will use GPU if available
        transformer_disable=True  # Disable transformer for simplified example
    )
    
    # Example text
    text = """
    Artificial intelligence is rapidly transforming various industries worldwide.
    Companies are adopting machine learning to automate tasks and gain insights.
    The technology promises increased efficiency and new capabilities.
    However, concerns about privacy and job displacement remain significant challenges.
    """
    
    print("\nOriginal text:")
    print(text)
    
    # Basic paraphrasing with default settings
    print("\n--- Default paraphrasing ---")
    paraphrased = paraphraser.paraphrase(
        text,
        rule_based_rate=0.4,  # Medium word replacement rate
        humanize=True,  # Apply humanization
        humanize_intensity=0.5  # Medium humanization intensity
    )
    print(paraphrased)
    
    # More aggressive paraphrasing
    print("\n--- Aggressive paraphrasing ---")
    paraphrased_aggressive = paraphraser.paraphrase(
        text,
        rule_based_rate=0.7,  # Higher word replacement rate
        humanize=True,
        humanize_intensity=0.7  # Higher humanization intensity
    )
    print(paraphrased_aggressive)
    
    # Only humanization, no word replacement
    print("\n--- Humanization only ---")
    humanized_only = paraphraser.paraphrase(
        text,
        rule_based_rate=0.0,  # No word replacement
        humanize=True,
        humanize_intensity=0.8  # Strong humanization
    )
    print(humanized_only)


if __name__ == "__main__":
    main()
