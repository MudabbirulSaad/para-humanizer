"""
Tests for the rule-based processor module.
"""
import unittest
from para_humanizer.processors.rule_based import RuleBasedProcessor


class TestRuleBasedProcessor(unittest.TestCase):
    """Test cases for rule-based text processing."""
    
    def setUp(self):
        """Set up a RuleBasedProcessor instance for testing."""
        self.processor = RuleBasedProcessor()
    
    def test_basic_word_replacement(self):
        """Test basic word replacement functionality."""
        text = "The quick fox jumps over the lazy dog."
        processed = self.processor.basic_word_replacement(text, replacement_rate=1.0)
        
        # The processed text should be different from the original
        self.assertNotEqual(text, processed)
        
        # The processed text should maintain similar length
        self.assertLess(abs(len(text) - len(processed)), len(text) * 0.5)
    
    def test_restructure_sentence(self):
        """Test sentence restructuring functionality."""
        sentence = "The cat sat on the mat."
        restructured = self.processor.restructure_sentence(sentence)
        
        # The restructured sentence should be different
        self.assertNotEqual(sentence, restructured)
        
        # Core words should still be present (semantic preservation)
        self.assertTrue("cat" in restructured.lower())
        self.assertTrue("mat" in restructured.lower())
    
    def test_combine_sentences(self):
        """Test sentence combination functionality."""
        sentences = [
            "The sun is shining.",
            "The birds are singing.",
            "It is a beautiful day."
        ]
        
        combined = self.processor.combine_sentences(sentences)
        
        # The result should be fewer sentences than the input
        self.assertLess(combined.count(". ") + combined.count("? ") + combined.count("! "), len(sentences))
        
        # Core content should be preserved
        self.assertTrue("sun" in combined.lower())
        self.assertTrue("birds" in combined.lower())
        self.assertTrue("beautiful" in combined.lower())


if __name__ == "__main__":
    unittest.main()
