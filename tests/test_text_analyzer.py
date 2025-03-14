"""
Unit tests for the Text Analyzer module.
"""
import pytest
from para_humanizer.utils.text_analyzer import TextAnalyzer


class TestTextAnalyzer:
    """Test cases for the TextAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()
    
    def test_analyze_returns_parameters(self):
        """Test that analyze returns a dictionary with all required parameters."""
        text = "This is a sample text for testing the text analyzer."
        params = self.analyzer.analyze(text)
        
        # Check that all required parameters are present
        assert "rule_based_rate" in params
        assert "transformer_rate" in params
        assert "humanize_intensity" in params
        assert "typo_rate" in params
        
        # Check parameter ranges
        assert 0.0 <= params["rule_based_rate"] <= 1.0
        assert 0.0 <= params["transformer_rate"] <= 1.0
        assert 0.0 <= params["humanize_intensity"] <= 1.0
        assert 0.0 <= params["typo_rate"] <= 1.0
    
    def test_analyze_different_text_types(self):
        """Test analysis of different text types produces different parameters."""
        formal_text = """
        The implementation of robust algorithmic solutions requires careful consideration of computational complexity.
        Furthermore, the development of efficient data structures is essential for optimizing performance in large-scale applications.
        Therefore, it is imperative that software engineers understand the theoretical underpinnings of computer science.
        """
        
        informal_text = """
        Hey! Just wanted to let you know that I'm running a bit late. Traffic is crazy today.
        Can you wait for me for like 10 mins? I'll buy you coffee when I get there :)
        """
        
        technical_text = """
        The system uses a distributed architecture with load balancing across multiple nodes.
        Data processing is handled by a pipeline that includes preprocessing, feature extraction, 
        and machine learning components for classification and prediction.
        """
        
        params_formal = self.analyzer.analyze(formal_text)
        params_informal = self.analyzer.analyze(informal_text)
        params_technical = self.analyzer.analyze(technical_text)
        
        # Print parameter values for debugging
        print(f"\nFormal text parameters: {params_formal}")
        print(f"Informal text parameters: {params_informal}")
        print(f"Technical text parameters: {params_technical}")
        
        # Formal text should have lower rule_based_rate and typo_rate
        assert params_formal["rule_based_rate"] < params_informal["rule_based_rate"], \
            f"Formal rule rate: {params_formal['rule_based_rate']} should be < Informal rule rate: {params_informal['rule_based_rate']}"
        assert params_formal["typo_rate"] < params_informal["typo_rate"], \
            f"Formal typo rate: {params_formal['typo_rate']} should be < Informal typo rate: {params_informal['typo_rate']}"
        
        # Technical text should have higher transformer_rate
        assert params_technical["transformer_rate"] >= params_informal["transformer_rate"], \
            f"Technical transformer rate: {params_technical['transformer_rate']} should be >= Informal transformer rate: {params_informal['transformer_rate']}"
    
    def test_analyze_empty_text(self):
        """Test handling of empty text."""
        params = self.analyzer.analyze("")
        
        # Should return default values for empty text
        assert params["rule_based_rate"] == 0.4
        assert params["transformer_rate"] == 0.0
        assert params["humanize_intensity"] == 0.5
        assert params["typo_rate"] == 0.01
    
    def test_calculate_metrics(self):
        """Test metric calculation for text analysis."""
        text = "This is a simple test. It has two sentences with some words."
        metrics = self.analyzer._calculate_metrics(text)
        
        # Check that all expected metrics are present
        assert "avg_sentence_length" in metrics
        assert "avg_word_length" in metrics
        assert "lexical_diversity" in metrics
        assert "punctuation_rate" in metrics
        assert "formality_level" in metrics
        assert "technical_level" in metrics
        assert "text_length" in metrics
        
        # Check specific values
        assert metrics["text_length"] == len(text)
        assert 0.0 <= metrics["lexical_diversity"] <= 1.0
        assert 0.0 <= metrics["punctuation_rate"] <= 1.0
