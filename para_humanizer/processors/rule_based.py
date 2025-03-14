"""
Rule-based text processors for UltimateParaphraser.
Provides classes and functions for synonym replacement and sentence restructuring.
"""
import nltk
from nltk.corpus import wordnet
import random
import spacy
import re
import threading
import os
from typing import List, Dict, Tuple, Optional, Set, Any, Union

from para_humanizer.utils.config_manager import get_config_manager
from para_humanizer.utils.synonym_loader import SynonymLibrary
from para_humanizer.utils.text_utils import is_protected_term

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If spaCy model isn't installed, download it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


class RuleBasedProcessor:
    """
    Implements rule-based methods for text paraphrasing including
    synonym replacement, sentence restructuring, and typo introduction.
    """
    
    def __init__(self, config_manager=None, synonym_library: Any = None, enable_learning: bool = True):
        """
        Initialize the rule-based processor with thread safety in mind.
        
        Args:
            config_manager: ConfigManager instance with configuration settings
            synonym_library: SynonymLibrary instance for synonym lookups
            enable_learning: Whether to enable automatic synonym learning
        """
        # Add lock for thread-safe WordNet access
        self.wordnet_lock = threading.Lock()
        
        # Get the configuration manager
        self.config_manager = config_manager if config_manager is not None else get_config_manager()
        
        # Use provided synonym library or create a default one
        if synonym_library is None:
            blacklist_words = set(self.config_manager.get_blacklist_words())
            common_words = set(self.config_manager.get_set('common_words'))
            from para_humanizer.utils.synonym_loader import get_synonym_library
            self.synonym_library = get_synonym_library(blacklist_words, common_words)
        else:
            self.synonym_library = synonym_library
        
        # Load configuration data
        self.common_words = set(self.config_manager.get_set('common_words'))
        self.protected_terms = set(self.config_manager.get_protected_terms())
        self.blacklist_words = set(self.config_manager.get_blacklist_words())
        
        # Load academic configuration
        self.academic_protected_terms = set(self.config_manager.get_set('academic_protected_terms'))
        self.academic_collocations = set(self.config_manager.get_set('academic_collocations'))
        self.academic_terms = set(self.config_manager.get_set('academic_terms'))
        self.academic_avoid = set(self.config_manager.get_set('academic_avoid'))
        self.avoid_suffixes = set(self.config_manager.get_set('avoid_suffixes'))
        self.avoid_prefixes = set(self.config_manager.get_set('avoid_prefixes'))
        self.avoid_patterns = set(self.config_manager.get_set('avoid_patterns'))
        
        self.tag_mapping = self.config_manager.get_tag_mapping()
        
        # Thread safety for WordNet access
        self.wordnet_lock = threading.Lock()
        
        # NLP model for dependency parsing
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except:
            print("Warning: Spacy model not available. Dependency parsing will be disabled.")
            self.nlp = None
        
        # Initialize synonym learner if enabled
        self.enable_learning = enable_learning
        self.synonym_learner = None
        
        if enable_learning:
            from para_humanizer.utils.synonym_learner import get_synonym_learner
            self.synonym_learner = get_synonym_learner(self.synonym_library)
        
        # Batch text for learning to improve efficiency
        self.learning_batch = []
        self.learning_batch_size = 10
        
    def get_wordnet_pos(self, nltk_tag: str) -> Optional[str]:
        """
        Convert NLTK POS tag to WordNet POS tag.
        
        Args:
            nltk_tag: The NLTK part-of-speech tag
            
        Returns:
            WordNet POS tag or None if no mapping exists
        """
        return self.tag_mapping.get(nltk_tag, None)
    
    def get_quality_synonyms(self, word: str, pos: str, context: str) -> List[str]:
        """
        Get appropriate synonyms with better filtering.
        
        Args:
            word: The word to find synonyms for
            pos: The part-of-speech tag (WordNet format)
            context: The surrounding text for context
            
        Returns:
            List of quality synonyms
        """
        word = word.lower()
        
        # Skip blacklisted words
        if word in self.blacklist_words:
            return []
            
        # First try the synonym library (preferred source)
        library_synonyms = self.synonym_library.get_synonyms(word, max_count=3)
        if library_synonyms:
            return library_synonyms
            
        # Adjust replacement probability for common words
        if word in self.common_words and random.random() < 0.7:
            return []
        
        # Use thread-safe access to WordNet as fallback
        synonyms = []
        try:
            with self.wordnet_lock:
                # Use WordNet with better filtering
                for syn in wordnet.synsets(word, pos=pos):
                    # Only consider the first few synsets (most common meanings)
                    if len(synonyms) >= 5:
                        break
                        
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        synonym_lower = synonym.lower()
                        
                        # Explicitly check against blacklist words
                        if synonym_lower in self.blacklist_words:
                            continue
                            
                        # Additional safety check for words containing inappropriate substrings
                        unsafe_substrings = ['cock', 'prick', 'dick', 'fuck', 'shit', 'ass', 'cum', 'tit']
                        if any(unsafe in synonym_lower for unsafe in unsafe_substrings):
                            continue
                        
                        # Apply quality filters
                        if (synonym != word and 
                            synonym not in synonyms and 
                            ' ' not in synonym and  # Single word only
                            '-' not in synonym and
                            len(synonym) >= 3 and
                            len(synonym) <= len(word) + 3 and  # Not much longer than original
                            not any(synonym.endswith(s) for s in ['ose', 'eous', 'ious', 'sis', 'ism']) and  # Skip uncommon suffixes
                            not synonym.startswith('un') and  # Skip negations
                            synonym.isalpha()):  # Only alphabetic characters
                            
                            # Check for common English words using basic frequency heuristic
                            if len(synonym) <= 8:
                                synonyms.append(synonym)
        except Exception as e:
            # If there's any error with WordNet, just return an empty list
            print(f"WordNet error for word '{word}': {str(e)}")
            return []
                        
        # Return up to 3 synonyms
        return synonyms[:3]
    
    def process_text(self, text: str, rate: float = 0.4, tone: str = "casual") -> str:
        """
        Process input text with rule-based synonym replacement.
        
        Args:
            text: The input text to process
            rate: The rate of word replacement (0.0 to 1.0)
            tone: The writing tone to use (casual, formal, academic)
            
        Returns:
            Processed text
        """
        if tone == "academic":
            return self.paraphrase_academic(text, rate)
        elif tone == "formal":
            return self.paraphrase_formal(text, rate)
        else:
            return self.paraphrase_text(text, rate)
    
    def paraphrase_academic(self, text: str, rate: float = 0.3) -> str:
        """
        Paraphrase academic text with specialized handling for technical and scientific terminology.
        Uses stricter synonym selection and a lower effective replacement rate.
        
        Args:
            text: The input text to paraphrase
            rate: The base rate of word replacement (will be adjusted downward)
            
        Returns:
            Academically-appropriate paraphrased text
        """
        # Use a more conservative replacement rate for academic text
        effective_rate = max(0.2, rate * 0.7)  # Reduce replacement rate
        
        # Tokenize the text
        sentences = nltk.sent_tokenize(text)
        result_sentences = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                result_sentences.append(sentence)
                continue
                
            # Tokenize and tag the words
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            
            # Initialize replacement array
            replaced_tokens = tokens.copy()
            
            # Generate a protection mask for academic and technical terms
            protected_mask = [False] * len(tokens)
            
            # Identify multi-word terms to protect
            for i in range(len(tokens)):
                # Check if this token starts a protected term
                if is_protected_term(tokens[i], sentence, self.protected_terms):
                    protected_mask[i] = True
                    continue
                
                # Check if this is part of an academic term (exact match)
                if tokens[i].lower() in self.academic_terms:
                    protected_mask[i] = True
                    continue
                
                # Protect capitalized words in academic contexts (likely proper nouns or specific terms)
                if len(tokens[i]) > 1 and tokens[i][0].isupper() and tokens[i][1:].islower():
                    protected_mask[i] = True
                    continue
            
            # Track replaced word count
            replaced_count = 0
            max_replacements = int(len(tokens) * effective_rate)
            tried_indices = set()
            
            # Attempt replacement until reaching the target rate
            while replaced_count < max_replacements and len(tried_indices) < len(tokens):
                # Choose a random word that hasn't been tried yet
                available_indices = [i for i in range(len(tokens)) if i not in tried_indices]
                if not available_indices:
                    break
                
                idx = random.choice(available_indices)
                tried_indices.add(idx)
                
                # Skip if this is a protected term
                if protected_mask[idx]:
                    continue
                    
                word = tokens[idx]
                tag = tagged[idx][1]
                
                # Skip short words, stopwords, non-content words
                if (len(word) < 4 or word.lower() in self.common_words or 
                    not word.isalpha() or word.lower() in self.blacklist_words):
                    continue
                
                # Get the WordNet POS tag
                wordnet_pos = self.get_wordnet_pos(tag)
                if not wordnet_pos:
                    continue
                
                # Determine if we should replace this word based on rate
                if random.random() < rate:
                    # Get context for better synonym selection
                    context_start = max(0, idx-3)
                    context_end = min(len(tokens), idx+4)
                    context = ' '.join(tokens[context_start:context_end])
                    
                    # Get quality synonyms and apply academic filtering
                    synonyms = self.get_academic_synonyms(word, wordnet_pos, context)
                    
                    if synonyms:
                        # Choose a synonym with preference for academic language
                        synonym = random.choice(synonyms)
                        
                        # Handle capitalization
                        if word[0].isupper():
                            synonym = synonym[0].upper() + synonym[1:]
                            
                        replaced_tokens[idx] = synonym
                        replaced_count += 1
            
            # Reconstruct the sentence
            result = ' '.join(replaced_tokens)
            
            # Ensure proper spacing around punctuation
            result = re.sub(r'\s+([.,;:!?)])', r'\1', result)
            result = re.sub(r'([(])\s+', r'\1', result)
            
            result_sentences.append(result)
            
        # Join processed sentences
        result_text = ' '.join(result_sentences)
        
        # Normalize whitespace
        result_text = re.sub(r'\s+', ' ', result_text)
        
        return result_text
    
    def get_academic_synonyms(self, word: str, pos: str, context: str) -> List[str]:
        """
        Get appropriate synonyms for academic text with stricter quality control.
        
        Args:
            word: The word to find synonyms for
            pos: The part-of-speech tag (WordNet format)
            context: The surrounding text for context
            
        Returns:
            List of quality academic synonyms
        """
        word = word.lower()
        
        # Skip blacklisted words and common words more aggressively
        if word in self.blacklist_words or word in self.common_words:
            return []
        
        # Context-based academic term detection
        # Check if the word appears in likely academic collocations
        for term in self.academic_collocations:
            if f"{word} {term}" in context or f"{term} {word}" in context:
                return []  # Protect words that form academic phrases
        
        # Skip additional academic words
        if word in self.academic_terms:
            return []
            
        # First try the synonym library (preferred source)
        library_synonyms = self.synonym_library.get_synonyms(word, max_count=3)
        
        # Filter library synonyms for academic appropriateness with higher standards
        if library_synonyms:
            academic_synonyms = [s for s in library_synonyms 
                                if (s.lower() not in self.academic_avoid and 
                                    not any(s.lower().endswith(suff) for suff in self.avoid_suffixes) and
                                    len(s) >= len(word) - 1)]  # Avoid very short synonyms that might change meaning
            if academic_synonyms:
                # Further filter: compare lengths to avoid very different words
                similar_length_synonyms = [s for s in academic_synonyms 
                                        if abs(len(s) - len(word)) <= max(2, len(word) // 4)]
                return similar_length_synonyms if similar_length_synonyms else academic_synonyms[:1]
        
        # Use thread-safe access to WordNet as fallback with strict academic filtering
        synonyms = []
        try:
            with self.wordnet_lock:
                # Use WordNet with stricter academic filtering
                for syn in wordnet.synsets(word, pos=pos):
                    # Only consider the first few synsets (most common meanings)
                    if len(synonyms) >= 2:  # Stricter limit for academic text
                        break
                        
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        synonym_lower = synonym.lower()
                        
                        # Apply academic quality filters
                        if (synonym != word and 
                            synonym_lower not in self.academic_avoid and
                            synonym not in synonyms and 
                            ' ' not in synonym and  # Single word only
                            '-' not in synonym and  # Avoid hyphenated terms
                            len(synonym) >= 3 and
                            len(synonym) >= len(word) - 1 and  # Not much shorter than original
                            len(synonym) <= len(word) + 2 and  # Not much longer than original
                            not any(synonym.endswith(s) for s in self.avoid_suffixes) and  # Avoid casual suffixes
                            not synonym.startswith(tuple(self.avoid_prefixes)) and  # Skip negations and modifiers
                            synonym.isalpha()):  # Only alphabetic characters
                            
                            # Frequency check - prefer words that are neither too common nor too rare
                            if not (synonym_lower in self.common_words):
                                # Additional check - make sure synonym doesn't appear in academic avoid patterns
                                if not any(avoid_term in synonym_lower for avoid_term in self.avoid_patterns):
                                    synonyms.append(synonym)
        except Exception as e:
            print(f"WordNet error: {e}")
            
        return synonyms
    
    def paraphrase_formal(self, text: str, rate: float = 0.35) -> str:
        """
        Paraphrase formal text with a slightly more conservative approach.
        
        Args:
            text: The input text to paraphrase
            rate: The rate of word replacement (0.0 to 1.0)
            
        Returns:
            Paraphrased text
        """
        # Tokenize the text
        sentences = nltk.sent_tokenize(text)
        result_sentences = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                result_sentences.append(sentence)
                continue
                
            # Tokenize and tag the words
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            
            paraphrased_words = []
            for i, (word, tag) in enumerate(tagged):
                # Skip punctuation and small words
                if tag in ['CC', 'DT', 'IN', '.', ',', ':', ';', "''", '""', '(', ')', '!', '?'] or len(word) < 4:
                    paraphrased_words.append(word)
                    continue
                    
                # Skip protected terms
                if is_protected_term(word, sentence, self.protected_terms):
                    paraphrased_words.append(word)
                    continue
                    
                # Get the WordNet POS
                wordnet_pos = self.get_wordnet_pos(tag)
                if not wordnet_pos:
                    paraphrased_words.append(word)
                    continue
                
                # Determine if we should replace this word based on rate
                if random.random() < rate:
                    # Get context for better synonym selection
                    context_start = max(0, i - 5)
                    context_end = min(len(tokens), i + 5)
                    context = ' '.join(tokens[context_start:context_end])
                    
                    # Get quality synonyms
                    synonyms = self.get_quality_synonyms(word, wordnet_pos, context)
                    
                    if synonyms:
                        # Choose a random synonym and maintain capitalization
                        synonym = random.choice(synonyms)
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        paraphrased_words.append(synonym)
                    else:
                        paraphrased_words.append(word)
                else:
                    paraphrased_words.append(word)
            
            # Reconstruct the sentence
            paraphrased_sentence = self._reconstruct_sentence(paraphrased_words)
            result_sentences.append(paraphrased_sentence)
            
        # Combine sentences
        paraphrased_text = ' '.join(result_sentences)
        
        # Provide feedback to the learner about successful paraphrasing
        if self.enable_learning and self.synonym_learner and paraphrased_text != text:
            self.synonym_learner.provide_feedback(paraphrased_text, success=True)
            
        return paraphrased_text
    
    def paraphrase_text(self, text: str, rate: float = 0.3, humanize: bool = False) -> str:
        """
        Paraphrase text using rule-based synonym replacement.
        
        Args:
            text: The input text to paraphrase
            rate: The rate of word replacement (0.0 to 1.0)
            humanize: Whether to apply humanization techniques
            
        Returns:
            Paraphrased text
        """
        # Submit text to learner for processing
        if self.enable_learning and self.synonym_learner:
            self.learning_batch.append(text)
            
            # Process in batches for efficiency
            if len(self.learning_batch) >= self.learning_batch_size:
                self._process_learning_batch()
        
        # Tokenize the text
        sentences = nltk.sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                paraphrased_sentences.append(sentence)
                continue
                
            # Tokenize and tag the words
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            
            paraphrased_words = []
            for i, (word, tag) in enumerate(pos_tags):
                # Skip punctuation and small words
                if tag in ['CC', 'DT', 'IN', '.', ',', ':', ';', "''", '""', '(', ')', '!', '?'] or len(word) < 4:
                    paraphrased_words.append(word)
                    continue
                    
                # Skip protected terms
                if is_protected_term(word, sentence, self.protected_terms):
                    paraphrased_words.append(word)
                    continue
                    
                # Get the WordNet POS
                wordnet_pos = self.get_wordnet_pos(tag)
                if not wordnet_pos:
                    paraphrased_words.append(word)
                    continue
                
                # Determine if we should replace this word based on rate
                if random.random() < rate:
                    # Get context for better synonym selection
                    context_start = max(0, i - 5)
                    context_end = min(len(words), i + 5)
                    context = ' '.join(words[context_start:context_end])
                    
                    # Get quality synonyms
                    synonyms = self.get_quality_synonyms(word, wordnet_pos, context)
                    
                    if synonyms:
                        # Choose a random synonym and maintain capitalization
                        synonym = random.choice(synonyms)
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        paraphrased_words.append(synonym)
                    else:
                        paraphrased_words.append(word)
                else:
                    paraphrased_words.append(word)
            
            # Reconstruct the sentence
            paraphrased_sentence = self._reconstruct_sentence(paraphrased_words)
            paraphrased_sentences.append(paraphrased_sentence)
            
        # Combine sentences
        paraphrased_text = ' '.join(paraphrased_sentences)
        
        # Provide feedback to the learner about successful paraphrasing
        if self.enable_learning and self.synonym_learner and paraphrased_text != text:
            self.synonym_learner.provide_feedback(paraphrased_text, success=True)
            
        return paraphrased_text
    
    def should_preserve_word(self, word: str, pos: str, text: str, academic: bool = False) -> bool:
        """
        Determine if a word should be preserved (not paraphrased).
        
        Args:
            word: Word to check
            pos: Part of speech tag
            text: Full text context
            academic: Whether this is academic text
            
        Returns:
            True if the word should be preserved, False otherwise
        """
        # Skip very short words as these are often particles or function words
        if len(word) <= 2:
            return True
            
        # Always preserve proper nouns
        if pos.startswith('NNP'):
            return True
            
        # Skip common words unless forced replacement
        word_lower = word.lower()
        if word_lower in self.common_words:
            return random.random() > 0.1  # Occasionally replace common words
            
        # Always preserve protected terms
        if is_protected_term(word, text, self.protected_terms) or word_lower in self.protected_terms:
            return True
            
        # For academic text, preserve academic terminology
        if academic:
            # Check whether word is in academic protected terms
            if word_lower in self.academic_protected_terms:
                return True
                
            # Check if word is part of a multiword protected term
            # This helps preserve terms like "quantum mechanics" even if we just see "quantum"
            for term in self.academic_protected_terms:
                if " " in term and word_lower in term.split():
                    if term in text.lower():
                        return True
                        
            # Special handling for domain-specific terms that should be preserved in academic writing
            if pos.startswith('N') and word_lower not in self.academic_avoid:
                # Preserve most nouns in academic text since they're likely field-specific terms
                if len(word) > 3 and word_lower not in self.academic_avoid:
                    return True
            
            # Protect stems of academic collocations
            for collocation in self.academic_collocations:
                parts = collocation.split()
                if word_lower in parts:
                    # If the full collocation appears in the text, protect this word
                    if collocation in text.lower():
                        return True
        
        return False
    
    def _process_learning_batch(self) -> None:
        """Process the accumulated batch of texts for learning."""
        if not self.enable_learning or not self.synonym_learner or not self.learning_batch:
            return
            
        # Create a combined text for better context
        combined_text = " ".join(self.learning_batch)
        
        # Process the combined text
        learned_synonyms = self.synonym_learner.process_text(combined_text)
        
        # Log any newly learned synonyms
        if learned_synonyms:
            learned_count = sum(len(syns) for syns in learned_synonyms.values())
            print(f"Learned {learned_count} new synonyms from batch of {len(self.learning_batch)} texts")
            
        # Clear the batch
        self.learning_batch = []
        
        # Save the learner state
        self.synonym_learner.save()
        
    def _reconstruct_sentence(self, words: List[str]) -> str:
        """
        Reconstruct a sentence from a list of words, handling punctuation correctly.
        
        Args:
            words: List of words to combine
            
        Returns:
            Reconstructed sentence
        """
        result = ""
        for i, word in enumerate(words):
            # Handle punctuation properly
            if word in ['.', ',', ':', ';', '!', '?', ')', ']', '}']:
                result = result.rstrip() + word + ' '
            elif i > 0 and words[i-1] in ['(', '[', '{']:
                result = result.rstrip() + word + ' '
            else:
                result += word + ' '
                
        return result.strip()
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the synonym learning process."""
        if self.enable_learning and self.synonym_learner:
            return self.synonym_learner.get_stats()
        return {"enabled": False}
        
    def reset_learning(self) -> None:
        """Reset the synonym learner."""
        if self.enable_learning and self.synonym_learner:
            self.synonym_learner.reset_stats()
            print("Synonym learner statistics reset")
    
    def introduce_typo(self, word: str, probability: float = 0.02) -> str:
        """
        Introduce realistic typos with better context awareness.
        
        Args:
            word: The word to potentially modify with a typo
            probability: Chance of introducing a typo (0.0 to 1.0)
            
        Returns:
            Word with typo or original word
        """
        # Skip typos for short words, non-alphabetic words, or if random check fails
        if len(word) <= 2 or not word.isalpha() or random.random() > probability:
            return word
            
        # If it's a common word with predefined typos, use those
        if word.lower() in self.common_typos:
            return random.choice(self.common_typos[word.lower()])
        
        # Otherwise, generate a realistic typo
        typo_type = random.randint(0, 3)
        
        if typo_type == 0 and len(word) > 3:  # Letter omission
            pos = random.randint(1, len(word) - 2)  # Don't remove first or last letter
            return word[:pos] + word[pos+1:]
            
        elif typo_type == 1 and len(word) > 2:  # Letter transposition
            pos = random.randint(0, len(word) - 2)
            return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            
        elif typo_type == 2:  # Letter substitution
            pos = random.randint(0, len(word) - 1)
            # Get adjacent letters on keyboard
            keyboard_adjacency = {
                'a': 'sqzw', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsrdf',
                'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn',
                'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
                'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
                'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
                'z': 'asx'
            }
            
            # Get a random adjacent key if available, otherwise a random letter
            if word[pos].lower() in keyboard_adjacency:
                adjacent_chars = keyboard_adjacency[word[pos].lower()]
                replacement = random.choice(adjacent_chars)
                # Preserve capitalization
                if word[pos].isupper():
                    replacement = replacement.upper()
            else:
                replacement = chr(random.randint(97, 122))  # Random lowercase letter
                if word[pos].isupper():
                    replacement = replacement.upper()
                    
            return word[:pos] + replacement + word[pos+1:]
            
        elif typo_type == 3 and len(word) > 1:  # Doubled letter
            pos = random.randint(0, len(word) - 1)
            return word[:pos] + word[pos] + word[pos:]
            
        # Default: return original word
        return word
    
    def maybe_fix_typo(self, word_with_typo: str, original_word: str, probability: float = 0.6) -> str:
        """
        Add realistic typo corrections the way humans do when chatting or writing emails.
        
        Args:
            word_with_typo: The misspelled word
            original_word: The correct word
            probability: Chance of fixing the typo (0.0 to 1.0)
            
        Returns:
            Word with correction markers or original typo
        """
        if random.random() > probability:
            return word_with_typo
            
        # Different correction styles
        style = random.randint(0, 3)
        
        if style == 0:  # Asterisk correction
            return f"{word_with_typo} *{original_word}*"
            
        elif style == 1:  # Parenthetical correction
            return f"{word_with_typo} ({original_word})"
            
        elif style == 2:  # "I mean" correction
            return f"{word_with_typo} I mean {original_word}"
            
        else:  # Direct replacement with slash
            return f"{word_with_typo}/{original_word}"
    
    def basic_word_replacement(self, text: str, replacement_rate: float = 0.3, typo_rate: float = 0.005) -> str:
        """
        Replace words with synonyms using improved selection.
        
        Args:
            text: The text to process
            replacement_rate: Rate of word replacement (0.0 to 1.0)
            typo_rate: Rate of introducing realistic typos (0.0 to 1.0)
            
        Returns:
            Text with replaced words and occasional typos
        """
        # Identify protected terms
        protected_words = set()
        for term in self.protected_terms:
            if term.lower() in text.lower():
                protected_words.update(term.split())
        
        # Tokenize and tag parts of speech with thread safety
        try:
            with self.wordnet_lock:
                tokens = nltk.word_tokenize(text)
                tagged = nltk.pos_tag(tokens)
        except Exception as e:
            print(f"Error in tokenization: {str(e)}")
            # Return the original text on error
            return text
        
        result = []
        for word, tag in tagged:
            word_lower = word.lower()
            
            # Skip protected words and blacklisted terms
            if (word_lower in protected_words or 
                is_protected_term(word, text, self.protected_terms) or
                word_lower in self.blacklist_words):
                result.append(word)
                continue
                
            # Skip short words, punctuation, proper nouns
            if (len(word) < 4 or 
                not word.isalnum() or 
                tag.startswith('NNP')):
                result.append(word)
                continue
                
            # Get the WordNet POS tag
            wordnet_pos = get_wordnet_pos(tag)
            if not wordnet_pos or random.random() > replacement_rate:
                # Consider introducing a typo instead
                result.append(self.introduce_typo(word, typo_rate))
                continue
                
            # Get better quality synonyms
            synonyms = self.get_quality_synonyms(word_lower, wordnet_pos, text)
            
            if synonyms:
                replacement = random.choice(synonyms)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                    
                # Occasionally add a typo to the replacement
                if random.random() < typo_rate:
                    original = replacement
                    replacement = self.introduce_typo(replacement, 0.8)  # Higher chance for introduced word
                    # Sometimes "fix" the typo
                    replacement = self.maybe_fix_typo(replacement, original, 0.3)
                    
                result.append(replacement)
            else:
                # Consider introducing a typo to the original word
                result.append(self.introduce_typo(word, typo_rate))
                
        # Join with spaces and fix punctuation
        raw_result = ' '.join(result)
        
        # Fix spaces before punctuation
        result_with_spaces_fixed = re.sub(r'\s+([.,;:!?)])', r'\1', raw_result)
        
        # Fix spacing around apostrophes (critical for contractions)
        result_with_apostrophes_fixed = re.sub(r'\s+\'', '\'', result_with_spaces_fixed)
        result_with_apostrophes_fixed = re.sub(r'\'\s+', '\'', result_with_apostrophes_fixed)
        
        return result_with_apostrophes_fixed
    
    def restructure_sentence(self, sentence: str) -> str:
        """
        Apply more advanced sentence restructuring techniques.
        
        Args:
            sentence: The sentence to restructure
            
        Returns:
            Restructured sentence or original if no changes applied
        """
        doc = nlp(sentence)
        
        # Skip very short sentences or those with complex structure
        if len(sentence.split()) < 5 or len(sentence.split()) > 25:
            return sentence
            
        # Randomly select a restructuring method
        method = random.randint(0, 4)
        
        # 1. Clause order inversion (when there's a comma)
        if method == 0 and ',' in sentence:
            parts = sentence.split(',', 1)
            if len(parts) == 2 and len(parts[0].split()) > 2 and len(parts[1].split()) > 2:
                # Make sure part after comma doesn't start with a conjunction
                conjunctions = ['and', 'but', 'so', 'which', 'where', 'when', 'while']
                second_part_start = parts[1].strip().split()[0].lower()
                
                if second_part_start not in conjunctions:
                    # Capitalize the second part
                    second_part = parts[1].strip()
                    if second_part[0].islower():
                        second_part = second_part[0].upper() + second_part[1:]
                        
                    # Add a fitting conjunction to the first part
                    first_part = parts[0].strip()
                    if first_part[0].isupper():
                        first_part = first_part[0].lower() + first_part[1:]
                    
                    conjunction = random.choice(['while', 'although', 'whereas', 'since'])
                    # 50% chance to place conjunction with first or second clause
                    if random.random() < 0.5:
                        return f"{second_part}, {conjunction} {first_part}."
                    else:
                        return f"{conjunction} {first_part}, {second_part}."
        
        # 2. Active to passive voice conversion (simple cases)
        elif method == 1:
            # Simple heuristic: look for subject-verb-object pattern
            subjects = [token for token in doc if token.dep_ in ("nsubj", "nsubjpass")]
            verbs = [token for token in doc if token.pos_ == "VERB"]
            objects = [token for token in doc if token.dep_ in ("dobj", "pobj")]
            
            if subjects and verbs and objects:
                # Get the main elements
                subj = subjects[0]
                verb = verbs[0]
                obj = objects[0]
                
                # Check if they appear in typical order
                if subj.i < verb.i and verb.i < obj.i:
                    # Convert to passive voice
                    before_subj = ' '.join([t.text for t in doc[:subj.i]])
                    passive_verb = f"is {verb.lemma_}ed by" if not verb.lemma_.endswith('e') else f"is {verb.lemma_}d by"
                    after_obj = ' '.join([t.text for t in doc[obj.i+1:]])
                    
                    return f"{before_subj}{obj.text} {passive_verb} {subj.text} {after_obj}"
        
        # 3. Split into two sentences
        elif method == 3 and len(sentence.split()) > 12:
            # Look for good split points
            words = sentence.split()
            middle = len(words) // 2
            
            # Try to find a good split point near the middle
            split_point = None
            for i in range(middle - 3, middle + 3):
                if i > 0 and i < len(words) - 1:
                    # Good candidates for split are after conjunctions, adverbs, etc.
                    if words[i].lower() in ["and", "but", "so", "which", "where", "when", "while"]:
                        split_point = i
                        break
                        
            if split_point:
                first_part = ' '.join(words[:split_point])
                second_part = ' '.join(words[split_point+1:])
                
                # Make sure first part ends with punctuation
                if not first_part.endswith(('.', '!', '?')):
                    first_part += '.'
                    
                # Capitalize second part
                if second_part and second_part[0].islower():
                    second_part = second_part[0].upper() + second_part[1:]
                    
                return f"{first_part} {second_part}"
        
        # 4. Add an explanatory phrase
        elif method == 4:
            words = sentence.split()
            
            # Only for medium-length sentences
            if 7 <= len(words) <= 20:
                # Insert point around 1/3 through the sentence
                insert_point = len(words) // 3
                
                # Create an explanatory phrase
                phrases = [
                    "in other words",
                    "that is to say",
                    "to put it differently",
                    "to be specific",
                    "to be precise",
                    "to clarify",
                    "specifically",
                    "namely",
                    "particularly"
                ]
                
                phrase = random.choice(phrases)
                words.insert(insert_point, f", {phrase},")
                
                return ' '.join(words)
                
        # Default: return original
        return sentence
    
    def combine_sentences(self, sentences: List[str]) -> List[str]:
        """
        Combine short, related sentences with natural transitions.
        
        Args:
            sentences: List of sentences to potentially combine
            
        Returns:
            List of sentences, potentially with some combined
        """
        if len(sentences) < 2:
            return sentences
            
        result = []
        i = 0
        
        while i < len(sentences):
            # If we're at the last sentence or the current sentence is long
            if i == len(sentences) - 1 or len(sentences[i].split()) > 10:
                result.append(sentences[i])
                i += 1
                continue
                
            # If both current and next sentences are short, consider combining
            if len(sentences[i].split()) <= 10 and len(sentences[i+1].split()) <= 10:
                # 30% chance to combine (reduced from previous versions)
                if random.random() > 0.7:
                    # Remove period from first sentence and lowercase second sentence start
                    first = sentences[i].rstrip('.!?')
                    second = sentences[i+1]
                    if second and second[0].isupper():
                        second = second[0].lower() + second[1:]
                    
                    # Choose an appropriate transition based on context
                    # Simple heuristic: look for keywords to determine relationship
                    
                    first_lower = first.lower()
                    second_lower = second.lower()
                    
                    # Contrast relationship
                    if any(word in second_lower for word in ["however", "but", "despite", "yet", "although"]):
                        transitions = [" but ", " yet ", " although ", " while ", " whereas "]
                    
                    # Causal relationship
                    elif any(word in second_lower for word in ["therefore", "thus", "so", "consequently", "because"]):
                        transitions = [" so ", " because ", " since ", " as ", " therefore "]
                    
                    # Addition relationship
                    elif any(word in second_lower for word in ["also", "additionally", "furthermore", "moreover"]):
                        transitions = [" and ", " also ", " additionally ", " plus ", " furthermore "]
                    
                    # Temporal relationship
                    elif any(word in second_lower for word in ["then", "after", "before", "when", "while"]):
                        transitions = [" after which ", " before ", " when ", " as ", " while "]
                    
                    # Default: general connections
                    else:
                        transitions = [
                            " and ", " while ", " as ", " since ", 
                            " which means that ", " and therefore ",
                            " which is why ", " and consequently "
                        ]
                    
                    transition = random.choice(transitions)
                    
                    # Remove redundant words after adding transition
                    # E.g., avoid "X and. Additionally Y" -> "X and additionally Y"
                    redundant_starts = ["also", "additionally", "furthermore", "however", 
                                      "but", "yet", "so", "therefore", "thus"]
                    
                    second_words = second.split()
                    if second_words and second_words[0].lower() in redundant_starts:
                        second = ' '.join(second_words[1:])
                    
                    # Combine with transition
                    combined = first + transition + second
                    result.append(combined)
                    i += 2
                else:
                    result.append(sentences[i])
                    i += 1
            else:
                result.append(sentences[i])
                i += 1
                
        return result
    
    def process_chunk(self, chunk: str, replacement_rate: float = 0.4, 
                     typo_rate: float = 0.005) -> str:
        """
        Process a single chunk with rule-based methods.
        
        Args:
            chunk: Text chunk to process
            replacement_rate: Rate of word replacement (0.0 to 1.0)
            typo_rate: Rate of introducing typos (0.0 to 1.0)
            
        Returns:
            Processed text
        """
        # Check for empty or invalid chunk
        if not chunk or not chunk.strip():
            return ""
            
        # Apply rule-based word replacement
        paraphrased = self.basic_word_replacement(
            chunk, replacement_rate=replacement_rate, typo_rate=typo_rate
        )
        
        # Apply sentence structure variations
        doc = nlp(paraphrased)
        sentences = [sent.text for sent in doc.sents]
        
        # Restructure some sentences
        restructured = []
        for sentence in sentences:
            # 30% chance to restructure a sentence
            if random.random() < 0.3:
                restructured.append(self.restructure_sentence(sentence))
            else:
                restructured.append(sentence)
        
        # Try to combine some sentences
        restructured = self.combine_sentences(restructured)
        paraphrased = ' '.join(restructured)
        
        return paraphrased

    def paraphrase_academic(self, text: str, rate: float = 0.3) -> str:
        """
        Paraphrase academic text with specialized handling for technical and scientific terminology.
        Uses stricter synonym selection and a lower effective replacement rate.
        
        Args:
            text: The input text to paraphrase
            rate: The base rate of word replacement (will be adjusted downward)
            
        Returns:
            Academically-appropriate paraphrased text
        """
        # Use a more conservative replacement rate for academic text
        effective_rate = max(0.2, rate * 0.7)  # Reduce replacement rate
        
        # Tokenize the text
        sentences = nltk.sent_tokenize(text)
        result_sentences = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                result_sentences.append(sentence)
                continue
                
            # Tokenize and tag the words
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            
            # Initialize replacement array
            replaced_tokens = tokens.copy()
            
            # Track replaced word count
            replaced_count = 0
            max_replacements = int(len(tokens) * effective_rate)
            tried_indices = set()
            
            # Attempt replacement until reaching the target rate
            while replaced_count < max_replacements and len(tried_indices) < len(tokens):
                # Choose a random word that hasn't been tried yet
                available_indices = [i for i in range(len(tokens)) if i not in tried_indices]
                if not available_indices:
                    break
                
                idx = random.choice(available_indices)
                tried_indices.add(idx)
                
                # Skip if this is a protected term
                word = tokens[idx]
                tag = tagged[idx][1]
                
                # Skip short words, stopwords, non-content words
                if (len(word) < 4 or word.lower() in self.common_words or 
                    not word.isalpha() or word.lower() in self.blacklist_words):
                    continue
                
                # Check if we should preserve this word based on academic context
                if self.should_preserve_word(word, tag, sentence, academic=True):
                    continue
                
                # Get the WordNet POS tag
                wordnet_pos = self.get_wordnet_pos(tag)
                if not wordnet_pos:
                    continue
                
                # Determine if we should replace this word based on rate
                if random.random() < rate:
                    # Get context for better synonym selection
                    context_start = max(0, idx-3)
                    context_end = min(len(tokens), idx+4)
                    context = ' '.join(tokens[context_start:context_end])
                    
                    # Get quality synonyms and apply academic filtering
                    synonyms = self.get_academic_synonyms(word, wordnet_pos, context)
                    
                    if synonyms:
                        # Choose a synonym with preference for academic language
                        synonym = random.choice(synonyms)
                        
                        # Handle capitalization
                        if word[0].isupper():
                            synonym = synonym[0].upper() + synonym[1:]
                            
                        replaced_tokens[idx] = synonym
                        replaced_count += 1
            
            # Reconstruct the sentence
            result = ' '.join(replaced_tokens)
            
            # Ensure proper spacing around punctuation
            result = re.sub(r'\s+([.,;:!?)])', r'\1', result)
            result = re.sub(r'([(])\s+', r'\1', result)
            
            result_sentences.append(result)
            
        # Join processed sentences
        result_text = ' '.join(result_sentences)
        
        # Normalize whitespace
        result_text = re.sub(r'\s+', ' ', result_text)
        
        return result_text
