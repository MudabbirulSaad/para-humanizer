"""
Text analyzer module for UltimateParaphraser.
Analyzes text to determine optimal paraphrasing parameters.
"""
import re
import nltk
import logging
from typing import Dict, Any, List, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Analyzes text characteristics to suggest optimal paraphrasing parameters.
    """
    
    def __init__(self):
        """Initialize the text analyzer with necessary resources."""
        # Ensure we have the necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text and suggest optimal paraphrasing parameters.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of suggested parameter values
        """
        # Handle empty text with default parameters
        if not text or text.strip() == "":
            logger.info("Empty text received, using default parameters")
            return {
                "rule_based_rate": 0.4,
                "transformer_rate": 0.0,
                "humanize_intensity": 0.5,
                "typo_rate": 0.01,
                "academic_text": False
            }
            
        # Get text metrics
        metrics = self._calculate_metrics(text)
        
        # Calculate parameters based on metrics
        params = self._calculate_parameters(metrics)
        
        logger.info(f"Analyzed text with {len(text)} characters, suggested parameters: {params}")
        return params
    
    def _calculate_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate various metrics about the input text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of metrics
        """
        # Tokenize the text
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Calculate basic metrics
        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        
        # Calculate lexical diversity (unique words / total words)
        unique_words = set(word.lower() for word in words if word.isalpha())
        lexical_diversity = len(unique_words) / max(1, len([w for w in words if w.isalpha()]))
        
        # Calculate punctuation frequency
        punctuation_count = sum(1 for char in text if char in '.,;:!?-()[]{}\'\"')
        punctuation_rate = punctuation_count / max(1, len(text))
        
        # Detect formality level (simple heuristic)
        formality_indicators = ['therefore', 'however', 'thus', 'consequently', 
                               'furthermore', 'nevertheless', 'moreover', 'hereby',
                               'whereas', 'accordingly', 'henceforth']
        formality_matches = sum(1 for word in words if word.lower() in formality_indicators)
        formality_level = min(1.0, formality_matches / max(10, len(sentences)))
        
        # Check for academic writing indicators
        academic_indicators = [
            'analyze', 'analysis', 'argue', 'argument', 'concept', 'conclude', 'conclusion',
            'data', 'define', 'definition', 'demonstrate', 'develop', 'emphasize', 'empirical',
            'establish', 'estimate', 'evaluate', 'evidence', 'examine', 'focus', 'framework',
            'hypothesis', 'identify', 'implement', 'implication', 'indicate', 'individual',
            'interpret', 'literature', 'methodology', 'obtain', 'participate', 'perceive',
            'procedure', 'process', 'research', 'resolve', 'resource', 'respond', 'role',
            'section', 'significant', 'similar', 'source', 'specific', 'structure', 'theory',
            'variable', 'whereas', 'abstract', 'investigation', 'subsequently', 'postulate',
            'paradigm', 'via', 'utilize', 'optimal', 'methodology', 'preliminary', 'phenomenon'
        ]
        academic_matches = sum(1 for word in words if word.lower() in academic_indicators)
        academic_level = min(1.0, academic_matches / max(15, len(sentences)))
        
        # Check for technical content
        technical_indicators = ['data', 'analysis', 'system', 'process', 'function',
                               'method', 'algorithm', 'implementation', 'interface',
                               'configuration', 'parameter', 'component', 'module']
        technical_matches = sum(1 for word in words if word.lower() in technical_indicators)
        technical_level = min(1.0, technical_matches / max(10, len(sentences)))
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity,
            'punctuation_rate': punctuation_rate,
            'formality_level': formality_level,
            'academic_level': academic_level,
            'technical_level': technical_level,
            'text_length': len(text)
        }
    
    def _calculate_parameters(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate optimal paraphrasing parameters based on text metrics.
        
        Args:
            metrics: Dictionary of text metrics
            
        Returns:
            Dictionary of suggested parameter values
        """
        # Base parameters
        params = {
            'rule_based_rate': 0.4,
            'transformer_rate': 0.0,
            'humanize_intensity': 0.5,
            'typo_rate': 0.01,
            'academic_text': False
        }
        
        # Adjust rule_based_rate based on lexical diversity and formality
        # Higher diversity and formality = lower rate to preserve style
        diversity_factor = 1.0 - (metrics['lexical_diversity'] * 0.5)
        formality_factor = 1.0 - (metrics['formality_level'] * 0.7)
        
        # For formal text, we want lower rule_based_rate to preserve style
        if metrics['formality_level'] > 0.3:
            params['rule_based_rate'] = max(0.2, 0.35 * diversity_factor * formality_factor)
        else:
            # For informal text, we want higher rule_based_rate for more variation
            params['rule_based_rate'] = min(0.7, max(0.4, 0.5 * diversity_factor))
        
        # Adjust transformer_rate based on technical content
        # More technical content = higher transformer rate for better paraphrasing
        if metrics['technical_level'] > 0.2:
            # For technical text, use higher transformer rate
            params['transformer_rate'] = min(0.3, max(0.1, metrics['technical_level'] * 0.5))
        else:
            # For non-technical text, keep transformer rate low
            params['transformer_rate'] = min(0.1, metrics['technical_level'] * 0.3)
        
        # Adjust humanize_intensity based on formality
        # More formal = less humanization
        humanize_factor = 1.0 - (metrics['formality_level'] * 0.6)
        params['humanize_intensity'] = max(0.2, min(0.7, 0.5 * humanize_factor))
        
        # Set typo_rate lower for formal content
        if metrics['formality_level'] > 0.3:
            params['typo_rate'] = max(0.0, min(0.005, 0.01 - metrics['formality_level'] * 0.01))
        
        # Detect academic text and set a flag
        params['academic_text'] = metrics['academic_level'] > 0.3 or metrics['formality_level'] > 0.5
        
        return params


# Create a singleton instance for use throughout the application
text_analyzer = TextAnalyzer()
