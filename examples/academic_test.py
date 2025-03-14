#!/usr/bin/env python
"""
Test script for demonstrating academic tone features in Para-Humanizer.
"""
import os
import sys
import time
from para_humanizer.core.paraphraser import UltimateParaphraser

def main():
    # Initialize the paraphraser
    print("Initializing UltimateParaphraser...\n")
    paraphraser = UltimateParaphraser(
        use_gpu=True,  # Will use GPU if available
        transformer_disable=True  # Disable transformer for simplified example
    )
    
    # Academic text example with scientific content
    academic_text = """
    The quantum mechanical model is a mathematical description of the behavior of electrons in atoms and molecules. 
    It incorporates the principles of quantum theory, which suggests that energy and matter can exhibit properties 
    of both particles and waves. This model supersedes the Bohr model by describing electrons as three-dimensional 
    wave functions rather than as particles moving in discrete orbits.
    
    In academic discourse, it is essential to maintain methodological rigor and provide substantial evidence for 
    claims. Researchers should clearly articulate their hypotheses and ensure that their conclusions are supported 
    by empirical data. The literature review section of a paper establishes the theoretical framework within which 
    the research is situated.
    """
    
    # Additional technical academic content
    technical_text = """
    The regression analysis yielded a correlation coefficient of r = 0.78 (p < 0.001), indicating a strong positive 
    relationship between the independent and dependent variables. The coefficient of determination (R²) was 0.61, 
    suggesting that 61% of the variance can be explained by the model. Factor analysis revealed three primary components 
    with eigenvalues exceeding 1.0, which together accounted for 74% of the total variance in the dataset.
    
    The methodology employed a mixed-methods approach, combining quantitative surveys (n=427) with qualitative 
    interviews (n=28). Statistical analysis was conducted using SPSS software version A multivariate analysis of 
    variance (MANOVA) was performed to examine the interaction effects between demographic variables and treatment conditions.
    """
    
    # Create output file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "academic_test_results.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ACADEMIC TONE TEST RESULTS\n")
        f.write("==========================\n\n")
        
        f.write("Original text (Scientific):\n")
        f.write(academic_text)
        f.write("\n" + "-"*80 + "\n\n")
        
        # Process with academic tone
        print("Processing scientific text with academic tone...")
        academic_result = paraphraser.paraphrase(
            text=academic_text,
            rule_based_rate=0.4,
            transformer_rate=0.0,
            humanize=True,
            humanize_intensity=0.3,
            typo_rate=0.0,
            no_parallel=False,
            preserve_structure=True,
            tone="academic"
        )
        f.write("ACADEMIC TONE RESULTS (Scientific):\n")
        f.write(academic_result)
        f.write("\n\n" + "-"*80 + "\n\n")
        
        # Process with casual tone
        print("Processing scientific text with casual tone...")
        casual_result = paraphraser.paraphrase(
            text=academic_text,
            rule_based_rate=0.4,
            transformer_rate=0.0,
            humanize=True,
            humanize_intensity=0.6,
            typo_rate=0.01,
            no_parallel=False,
            preserve_structure=True,
            tone="casual"
        )
        f.write("CASUAL TONE RESULTS (Scientific):\n")
        f.write(casual_result)
        f.write("\n\n" + "-"*80 + "\n\n")
        
        # Process with formal tone
        print("Processing scientific text with formal tone...")
        formal_result = paraphraser.paraphrase(
            text=academic_text,
            rule_based_rate=0.4,
            transformer_rate=0.0,
            humanize=True,
            humanize_intensity=0.4,
            typo_rate=0.0,
            no_parallel=False,
            preserve_structure=True,
            tone="formal"
        )
        f.write("FORMAL TONE RESULTS (Scientific):\n")
        f.write(formal_result)
        f.write("\n\n" + "-"*80 + "\n\n")
        
        # Now process the technical statistics text
        f.write("Original text (Statistical):\n")
        f.write(technical_text)
        f.write("\n" + "-"*80 + "\n\n")
        
        # Process technical text with academic tone
        print("Processing statistical text with academic tone...")
        tech_academic_result = paraphraser.paraphrase(
            text=technical_text,
            rule_based_rate=0.4,
            transformer_rate=0.0,
            humanize=True,
            humanize_intensity=0.3,
            typo_rate=0.0,
            no_parallel=False,
            preserve_structure=True,
            tone="academic"
        )
        f.write("ACADEMIC TONE RESULTS (Statistical):\n")
        f.write(tech_academic_result)
        f.write("\n\n" + "-"*80 + "\n\n")
        
        # Process technical text with casual tone for comparison
        print("Processing statistical text with casual tone...")
        tech_casual_result = paraphraser.paraphrase(
            text=technical_text,
            rule_based_rate=0.4,
            transformer_rate=0.0,
            humanize=True,
            humanize_intensity=0.6,
            typo_rate=0.01,
            no_parallel=False,
            preserve_structure=True,
            tone="casual"
        )
        f.write("CASUAL TONE RESULTS (Statistical):\n")
        f.write(tech_casual_result)
    
    print(f"\nResults have been written to: {output_file}")

if __name__ == "__main__":
    main()
