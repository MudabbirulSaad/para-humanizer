"""
Tests for the core paraphraser module.
"""
import unittest
from para_humanizer.core.paraphraser import UltimateParaphraser


class TestUltimateParaphraser(unittest.TestCase):
    """Test cases for the UltimateParaphraser class."""
    
    def setUp(self):
        """Set up a UltimateParaphraser instance for testing."""
        # Use CPU and disable transformer to speed up tests
        self.paraphraser = UltimateParaphraser(
            use_gpu=False, 
            transformer_disable=True
        )
    
    def test_initialization(self):
        """Test paraphraser initialization."""
        self.assertEqual(self.paraphraser.device, "cpu")
        self.assertTrue(self.paraphraser.transformer_disable)
    
    def test_basic_paraphrase(self):
        """Test basic paraphrasing functionality."""
        text = "The quick brown fox jumps over the lazy dog."
        
        # Test with default settings
        paraphrased = self.paraphraser.paraphrase(
            text,
            rule_based_rate=0.4,
            humanize=True,
            humanize_intensity=0.5
        )
        
        # The paraphrased text should be different
        self.assertNotEqual(text, paraphrased)
        
        # The paraphrased text should have similar length
        self.assertLess(abs(len(text) - len(paraphrased)), len(text) * 0.5)
    
    def test_humanize_only(self):
        """Test humanize-only functionality."""
        text = "This is a test sentence for humanization only."
        
        # Apply only humanization without other paraphrasing
        humanized = self.paraphraser.paraphrase(
            text,
            rule_based_rate=0.0,
            humanize=True,
            humanize_intensity=0.7
        )
        
        # The humanized text should be different
        self.assertNotEqual(text, humanized)
        
        # Core content should be preserved
        self.assertTrue("test" in humanized.lower())
        self.assertTrue("sentence" in humanized.lower())
    
    def test_no_modifications(self):
        """Test with all modifications disabled."""
        text = "This text should remain unchanged."
        
        # Disable all modifications
        unchanged = self.paraphraser.paraphrase(
            text,
            rule_based_rate=0.0,
            humanize=False,
            typo_rate=0.0
        )
        
        # The text should remain unchanged
        self.assertEqual(text, unchanged)


if __name__ == "__main__":
    unittest.main()
