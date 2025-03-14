"""
Tests for the humanizer processor module.
"""
import unittest
from para_humanizer.processors.humanizer import HumanizerProcessor


class TestHumanizerProcessor(unittest.TestCase):
    """Test cases for humanizer text processing."""
    
    def setUp(self):
        """Set up a HumanizerProcessor instance for testing."""
        self.processor = HumanizerProcessor()
    
    def test_add_fillers(self):
        """Test adding filler words to text."""
        text = "I think this is a good idea."
        humanized = self.processor.add_fillers(text, probability=1.0)
        
        # The humanized text should be longer
        self.assertGreater(len(humanized), len(text))
        
        # Original content should be preserved
        self.assertTrue("good idea" in humanized)
    
    def test_add_self_corrections(self):
        """Test adding self-corrections to text."""
        text = "The meeting is at 3 PM."
        humanized = self.processor.add_self_corrections(text, probability=1.0)
        
        # The humanized text should contain correction markers
        self.assertTrue("*" in humanized or "I mean" in humanized or "actually" in humanized)
        
        # Original content should be preserved
        self.assertTrue("meeting" in humanized)
        self.assertTrue("3 PM" in humanized)
    
    def test_apply_contractions(self):
        """Test applying contractions to text."""
        text = "I am not going to tell you that."
        humanized = self.processor.apply_contractions(text, probability=1.0)
        
        # The humanized text should contain contractions
        self.assertTrue("I'm" in humanized or "don't" in humanized or "won't" in humanized)
        
        # The humanized text should be shorter
        self.assertLess(len(humanized), len(text))
    
    def test_full_humanize(self):
        """Test the complete humanize process."""
        text = "This is a test sentence. We are checking the humanization process."
        humanized = self.processor.humanize(text, intensity=0.8)
        
        # The humanized text should be different from the original
        self.assertNotEqual(text, humanized)
        
        # Core content should be preserved
        self.assertTrue("test" in humanized.lower())
        self.assertTrue("humanization" in humanized.lower() or "checking" in humanized.lower())


if __name__ == "__main__":
    unittest.main()
