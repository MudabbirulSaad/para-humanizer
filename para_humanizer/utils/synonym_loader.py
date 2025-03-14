"""
Synonym loader for Para-Humanizer.
Provides utilities for loading and accessing synonyms from a JSON library.
"""
import json
import os
import logging
from typing import Dict, List, Optional, Set, Union
import random

# Setup logging
logger = logging.getLogger(__name__)

class SynonymLibrary:
    """
    Manages a comprehensive library of synonyms loaded from a JSON file.
    Provides methods to look up synonyms with category awareness.
    """
    
    def __init__(self, blacklist_words: Set[str], common_words: Set[str]):
        """
        Initialize the synonym library.
        
        Args:
            blacklist_words: Set of words that should never be used as synonyms
            common_words: Set of common words that may not need synonyms as often
        """
        self.synonyms: Dict[str, Dict[str, List[str]]] = {}
        self.word_to_category: Dict[str, str] = {}
        self.blacklist_words = blacklist_words
        self.common_words = common_words
        self.loaded = False
        
    def load_synonyms(self, filepath: str) -> bool:
        """
        Load synonyms from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing synonyms
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Synonym file not found: {filepath}")
                return False
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Skip meta information
            if 'meta' in data:
                data.pop('meta')
                
            # Process each category
            self.synonyms = {}
            self.word_to_category = {}
            
            for category, words in data.items():
                self.synonyms[category] = {}
                
                for word, synonyms in words.items():
                    # Filter out blacklisted words and duplicates
                    filtered_synonyms = []
                    for synonym in synonyms:
                        if synonym.lower() not in self.blacklist_words and synonym not in filtered_synonyms:
                            filtered_synonyms.append(synonym)
                    
                    self.synonyms[category][word] = filtered_synonyms
                    self.word_to_category[word] = category
                    
            self.loaded = True
            logger.info(f"Loaded synonym library with {len(self.word_to_category)} words "
                       f"across {len(self.synonyms)} categories")
            return True
            
        except Exception as e:
            logger.error(f"Error loading synonyms: {str(e)}")
            return False
            
    def get_synonyms(self, word: str, max_count: int = 3) -> List[str]:
        """
        Get synonyms for a word, with category awareness.
        
        Args:
            word: The word to find synonyms for
            max_count: Maximum number of synonyms to return
            
        Returns:
            List of synonyms for the word
        """
        word = word.lower()
        
        # Skip blacklisted and common words with some probability
        if word in self.blacklist_words:
            return []
            
        if word in self.common_words and random.random() < 0.7:
            return []
            
        # Check if we know the exact word
        for category, words in self.synonyms.items():
            if word in words:
                # Return random subset of synonyms for variety
                synonyms = words[word].copy()
                if len(synonyms) > max_count:
                    return random.sample(synonyms, max_count)
                return synonyms
                
        # If word not found directly, try case-insensitive match
        word_lower = word.lower()
        for category, words in self.synonyms.items():
            for w, syns in words.items():
                if w.lower() == word_lower:
                    if len(syns) > max_count:
                        return random.sample(syns, max_count)
                    return syns
                    
        # Word not found in our library
        return []
        
    def get_all_synonyms(self) -> Dict[str, List[str]]:
        """
        Get a flattened dictionary of all word -> synonyms mappings.
        
        Returns:
            Dictionary mapping words to their synonyms
        """
        result = {}
        for category, words in self.synonyms.items():
            for word, synonyms in words.items():
                result[word] = synonyms
        return result
        
    def add_synonym(self, word: str, synonym: str, category: Optional[str] = None) -> bool:
        """
        Add a new synonym to the library.
        
        Args:
            word: The word to add a synonym for
            synonym: The synonym to add
            category: Optional category (if not provided, will use existing category or 'misc')
            
        Returns:
            True if successful, False otherwise
        """
        if not self.loaded:
            logger.error("Cannot add synonym, library not loaded")
            return False
            
        # Skip blacklisted words
        if word.lower() in self.blacklist_words or synonym.lower() in self.blacklist_words:
            return False
            
        # Determine category
        if category is None:
            if word in self.word_to_category:
                category = self.word_to_category[word]
            else:
                category = "misc"
                
        # Ensure category exists
        if category not in self.synonyms:
            self.synonyms[category] = {}
            
        # Add or update word in category
        if word not in self.synonyms[category]:
            self.synonyms[category][word] = []
            self.word_to_category[word] = category
            
        # Add synonym if not already present
        if synonym not in self.synonyms[category][word]:
            self.synonyms[category][word].append(synonym)
            return True
            
        return False
        
    def save_synonyms(self, filepath: str) -> bool:
        """
        Save the current synonym library to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output = {
                "meta": {
                    "version": "1.1.0",
                    "description": "Curated synonym dictionary for Para-Humanizer",
                    "date_updated": "2025-03-14"
                }
            }
            
            # Add all categories and synonyms
            for category, words in self.synonyms.items():
                output[category] = words
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
                
            logger.info(f"Saved synonym library to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving synonyms: {str(e)}")
            return False


# Default synonym library path
DEFAULT_SYNONYM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources",
    "synonyms.json"
)

# Helper function to get an initialized synonym library
def get_synonym_library(blacklist_words: Optional[Set[str]] = None, 
                       common_words: Optional[Set[str]] = None, 
                       filepath: Optional[str] = None) -> SynonymLibrary:
    """
    Create and load a synonym library.
    
    Args:
        blacklist_words: Set of words that should never be used as synonyms
        common_words: Set of common words that may not need synonyms as often
        filepath: Optional custom path to synonym JSON file
        
    Returns:
        Loaded SynonymLibrary object
    """
    # Use ConfigManager if available
    try:
        from para_humanizer.utils.config_manager import get_config_manager
        config_manager = get_config_manager()
        
        if blacklist_words is None:
            blacklist_words = config_manager.get_blacklist_words()
            
        if common_words is None:
            common_words = config_manager.get_set("default.common_words")
    except ImportError:
        # Fallback to default empty sets if config_manager not available
        if blacklist_words is None:
            blacklist_words = set()
        if common_words is None:
            common_words = set()
        logger.warning("ConfigManager not available. Using fallback empty sets for blacklist and common words.")
    
    library = SynonymLibrary(blacklist_words, common_words)
    path = filepath or DEFAULT_SYNONYM_PATH
    
    if not library.load_synonyms(path):
        logger.warning(f"Failed to load synonyms from {path}. Using minimal functionality.")
    
    return library
