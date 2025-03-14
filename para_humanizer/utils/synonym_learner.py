"""
Automatic Synonym Learner for Para-Humanizer.
Provides functionality to discover and learn new synonyms from text.
"""
import os
import json
import logging
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import re

import nltk
from nltk.corpus import wordnet
import numpy as np

# Conditionally import spacy to manage dependencies gracefully
try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Try to import word vectors - if not available, we'll use a fallback approach
try:
    import gensim.downloader as gensim_downloader
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False

from para_humanizer.utils.config import TAG_MAPPING
from para_humanizer.utils.config_manager import get_config_manager
from para_humanizer.utils.synonym_loader import SynonymLibrary, DEFAULT_SYNONYM_PATH

# Setup logging
logger = logging.getLogger(__name__)

class SynonymLearner:
    """
    A class that learns new synonyms from text and context.
    Uses multiple strategies including word embeddings, context analysis,
    and WordNet expansion with a self-improving feedback mechanism.
    """
    
    def __init__(self, 
                 synonym_library: SynonymLibrary,
                 embedding_model: str = "glove-wiki-gigaword-100",
                 min_similarity: float = 0.70,
                 confidence_threshold: float = 0.65,
                 max_candidates_per_word: int = 5,
                 learning_rate: float = 0.05,
                 use_wordnet: bool = True):
        """
        Initialize the synonym learner with configuration parameters.
        
        Args:
            synonym_library: Existing SynonymLibrary to update
            embedding_model: Name of the word embedding model to use
            min_similarity: Minimum cosine similarity for word embeddings
            confidence_threshold: Minimum confidence to add a synonym
            max_candidates_per_word: Maximum number of candidates to consider
            learning_rate: Rate at which to update confidence scores
            use_wordnet: Whether to use WordNet as an additional source
        """
        self.synonym_library = synonym_library
        self.min_similarity = min_similarity
        self.confidence_threshold = confidence_threshold
        self.max_candidates_per_word = max_candidates_per_word
        self.learning_rate = learning_rate
        self.use_wordnet = use_wordnet
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Get blacklist and common words from config manager
        self.blacklist_words = self.config_manager.get_blacklist_words()
        self.common_words = self.config_manager.get_set("default.common_words")
        
        # Statistics for learned synonyms
        self.synonym_stats = defaultdict(lambda: defaultdict(dict))
        self.total_processed = 0
        self.total_learned = 0
        
        # Load embedding model if available
        self.word_vectors = None
        self.nlp = None
        self._load_dependencies(embedding_model)
        
        # Load spaCy for context analysis if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for context analysis")
            except OSError:
                logger.warning("Could not load spaCy model. Context analysis will be limited.")
                self.nlp = None
        
        # Path to store learned synonym stats
        self.stats_path = os.path.join(
            os.path.dirname(DEFAULT_SYNONYM_PATH),
            "synonym_stats.json"
        )
        
        # Load existing stats if available
        self._load_stats()
        
    def _load_dependencies(self, embedding_model: str) -> None:
        """Load word embeddings for similarity calculations."""
        if not WORD2VEC_AVAILABLE:
            logger.warning("Gensim not available. Word embedding similarity will not be used.")
            return
            
        try:
            logger.info(f"Loading word embedding model: {embedding_model}")
            self.word_vectors = gensim_downloader.load(embedding_model)
            logger.info("Word embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load word embedding model: {str(e)}")
            self.word_vectors = None
            
    def _load_stats(self) -> None:
        """Load previously saved synonym learning statistics."""
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert the JSON to our defaultdict structure
                for word, synonyms in data.get("stats", {}).items():
                    for synonym, stats in synonyms.items():
                        self.synonym_stats[word][synonym] = stats
                        
                self.total_processed = data.get("total_processed", 0)
                self.total_learned = data.get("total_learned", 0)
                
                logger.info(f"Loaded stats for {len(self.synonym_stats)} words")
            except Exception as e:
                logger.error(f"Failed to load synonym stats: {str(e)}")
        else:
            logger.info("No previous synonym stats found. Starting fresh.")
            
    def _save_stats(self) -> None:
        """Save synonym learning statistics to disk."""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            stats_dict = {}
            for word, synonyms in self.synonym_stats.items():
                stats_dict[word] = {}
                for synonym, stats in synonyms.items():
                    stats_dict[word][synonym] = stats
                    
            data = {
                "stats": stats_dict,
                "total_processed": self.total_processed,
                "total_learned": self.total_learned,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved synonym stats to {self.stats_path}")
        except Exception as e:
            logger.error(f"Failed to save synonym stats: {str(e)}")
            
    def _get_word_category(self, word: str, pos: str):
        """Determine the appropriate category for a word based on POS tag."""
        if word in self.synonym_library.word_to_category:
            return self.synonym_library.word_to_category[word]
            
        # Map POS tag to category
        if pos is not None:
            if pos.startswith('N'):
                return "nouns"
            elif pos.startswith('V'):
                return "verbs"
            elif pos.startswith('J'):
                return "adjectives"
            elif pos.startswith('R'):
                return "adverbs"
        
        # Check for technical terms using some heuristics
        if (len(word) > 8 or
            '_' in word or
            any(tech in word for tech in ['data', 'code', 'tech', 'api', 'web', 'cloud'])):
            return "technical_terms"
            
        # Default to nouns as most common
        return "nouns"
            
    def _embedding_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between word embeddings."""
        if not self.word_vectors:
            return 0.0
            
        try:
            return self.word_vectors.similarity(word1, word2)
        except (KeyError, ValueError):
            return 0.0
            
    def _get_wordnet_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Get synonyms from WordNet with filtering."""
        if not self.use_wordnet:
            return []
            
        # Convert NLTK POS to WordNet POS if provided
        wordnet_pos = None
        if pos:
            wordnet_pos = TAG_MAPPING.get(pos)
            
        synonyms = set()
        
        try:
            # Get synsets from WordNet
            synsets = wordnet.synsets(word, pos=wordnet_pos) if wordnet_pos else wordnet.synsets(word)
            
            # Extract lemma names from the first few synsets
            for synset in synsets[:3]:  # Limit to first 3 synsets for relevance
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    
                    # Apply some basic filtering
                    if (synonym != word and 
                        synonym.lower() not in self.blacklist_words and
                        ' ' not in synonym and
                        '-' not in synonym and
                        len(synonym) >= 3 and
                        synonym.isalpha()):
                        synonyms.add(synonym)
                        
            return list(synonyms)
        except Exception as e:
            logger.debug(f"WordNet error for {word}: {str(e)}")
            return []
            
    def _find_candidates_from_context(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find potential synonym candidates from text context.
        Uses NLP to identify words that appear in similar contexts.
        """
        if not self.nlp:
            return {}
            
        # Process the text with spaCy
        try:
            doc = self.nlp(text)
        except Exception as e:
            logger.error(f"Error processing text with spaCy: {str(e)}")
            return {}
            
        # Find potential synonym pairs based on context
        candidates = defaultdict(list)
        
        # Extract words with their context
        content_words = [token for token in doc if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV') 
                        and not token.is_stop and len(token.text) > 3]
        
        # Process each content word
        for token in content_words:
            word = token.text.lower()
            
            # Skip blacklisted words
            if word in self.blacklist_words:
                continue
                
            # Get context words (words that appear near this word)
            context_start = max(0, token.i - 5)
            context_end = min(len(doc), token.i + 5)
            context = [doc[i].text.lower() for i in range(context_start, context_end) 
                      if i != token.i and not doc[i].is_stop]
            
            # Check other content words with similar contexts
            for other_token in content_words:
                if other_token.i == token.i:
                    continue
                    
                other_word = other_token.text.lower()
                
                # Skip if words are not same POS
                if other_token.pos_ != token.pos_:
                    continue
                    
                # Get context for other word
                other_context_start = max(0, other_token.i - 5)
                other_context_end = min(len(doc), other_token.i + 5)
                other_context = [doc[i].text.lower() for i in range(other_context_start, other_context_end) 
                               if i != other_token.i and not doc[i].is_stop]
                
                # Calculate context overlap
                context_overlap = len(set(context) & set(other_context)) / max(1, len(set(context) | set(other_context)))
                
                # Calculate word embedding similarity
                embed_sim = self._embedding_similarity(word, other_word)
                
                # Calculate overall confidence score
                confidence = (0.6 * context_overlap + 0.4 * embed_sim)
                
                # Add as candidate if confidence is reasonable
                if confidence > 0.4:
                    candidates[word].append((other_word, confidence))
                    
        # Sort candidates by confidence and limit to top matches
        for word in candidates:
            candidates[word] = sorted(candidates[word], key=lambda x: x[1], reverse=True)
            candidates[word] = candidates[word][:self.max_candidates_per_word]
            
        return candidates
            
    def process_text(self, text: str, feedback: bool = False) -> Dict[str, List[Tuple[str, float]]]:
        """
        Process text to learn potential synonyms.
        
        Args:
            text: The text to process
            feedback: Whether this is feedback from successful paraphrasing
            
        Returns:
            Dictionary of learned synonyms with confidence scores
        """
        self.total_processed += 1
        
        # Clean and normalize text
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Skip very short texts
        if len(text.split()) < 10:
            return {}
            
        # Process with NLTK for POS tagging
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Find synonym candidates from context
        context_candidates = self._find_candidates_from_context(text)
        
        # Build final candidates combining multiple sources
        final_candidates = defaultdict(list)
        
        # Process each word
        for word, pos in pos_tags:
            word = word.lower()
            
            # Skip short words, stopwords, non-alphabetic words
            if (len(word) < 4 or word in self.blacklist_words or word in self.common_words or
                not word.isalpha() or word.isupper()):
                continue
                
            # Get category for this word
            category = self._get_word_category(word, pos)
            
            # Gather candidates from different sources
            candidates = []
            
            # 1. Add candidates from context analysis
            if word in context_candidates:
                candidates.extend(context_candidates[word])
                
            # 2. Add candidates from WordNet
            wordnet_synonyms = self._get_wordnet_synonyms(word, pos)
            for synonym in wordnet_synonyms:
                # Calculate confidence for WordNet synonyms
                # Exact WordNet matches get a baseline confidence of 0.6
                confidence = 0.6
                
                # Boost confidence if there's also embedding similarity
                embed_sim = self._embedding_similarity(word, synonym)
                if embed_sim > 0.5:
                    confidence += 0.1 * embed_sim
                    
                # Increase confidence if this is from feedback
                if feedback:
                    confidence += 0.1
                    
                candidates.append((synonym, confidence))
                
            # 3. Add candidates from word embeddings if available
            if self.word_vectors and word in self.word_vectors:
                try:
                    similar_words = self.word_vectors.most_similar(word, topn=10)
                    for similar_word, similarity in similar_words:
                        # Apply filtering
                        if (similar_word.lower() not in self.blacklist_words and
                            similar_word.isalpha() and len(similar_word) >= 3 and
                            similarity > self.min_similarity):
                            candidates.append((similar_word, similarity * 0.8))  # Scale down embedding confidence
                except Exception:
                    # Word might not be in vocabulary
                    pass
                    
            # Combine and deduplicate candidates
            seen = set()
            unique_candidates = []
            for synonym, confidence in candidates:
                if synonym not in seen and synonym != word:
                    seen.add(synonym)
                    
                    # Adjust confidence based on feedback
                    if feedback and (word in self.synonym_stats and 
                                    synonym in self.synonym_stats[word]):
                        # Positive reinforcement from feedback
                        prev_confidence = self.synonym_stats[word][synonym].get('confidence', 0.0)
                        confidence = prev_confidence + self.learning_rate * (1.0 - prev_confidence)
                        
                    unique_candidates.append((synonym, confidence))
                    
            # Sort by confidence and limit
            unique_candidates.sort(key=lambda x: x[1], reverse=True)
            final_candidates[word] = unique_candidates[:self.max_candidates_per_word]
            
        # Update stats and add high-confidence synonyms to library
        learned_synonyms = defaultdict(list)
        
        for word, candidates in final_candidates.items():
            category = self._get_word_category(word, None)  # Get category for the word
            
            for synonym, confidence in candidates:
                # Skip if confidence is too low
                if confidence < self.confidence_threshold:
                    continue
                    
                # Update stats for this word-synonym pair
                if synonym not in self.synonym_stats[word]:
                    self.synonym_stats[word][synonym] = {
                        'first_seen': time.strftime("%Y-%m-%d"),
                        'count': 1,
                        'confidence': confidence
                    }
                else:
                    # Update existing stats
                    self.synonym_stats[word][synonym]['count'] += 1
                    
                    # Exponential moving average for confidence
                    old_conf = self.synonym_stats[word][synonym]['confidence']
                    new_conf = old_conf * 0.8 + confidence * 0.2
                    self.synonym_stats[word][synonym]['confidence'] = new_conf
                    
                    # Use the updated confidence
                    confidence = new_conf
                    
                # Add to library if confidence is high enough
                if confidence >= self.confidence_threshold:
                    # Add to library
                    if word not in self.blacklist_words and synonym not in self.blacklist_words:
                        self.synonym_library.add_synonym(word, synonym, category)
                        learned_synonyms[word].append((synonym, confidence))
                        self.total_learned += 1
                        
        # Save stats periodically (every 10 processed texts)
        if self.total_processed % 10 == 0:
            self._save_stats()
            
        return learned_synonyms
        
    def reset_stats(self) -> None:
        """Reset learning statistics."""
        self.synonym_stats = defaultdict(lambda: defaultdict(dict))
        self.total_processed = 0
        self.total_learned = 0
        self._save_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about synonym learning."""
        return {
            "total_processed": self.total_processed,
            "total_learned": self.total_learned,
            "unique_words": len(self.synonym_stats),
            "unique_synonyms": sum(len(synonyms) for synonyms in self.synonym_stats.values()),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def provide_feedback(self, text: str, success: bool = True) -> None:
        """
        Provide feedback about paraphrasing quality to improve learning.
        
        Args:
            text: The successfully paraphrased text
            success: Whether the paraphrasing was successful
        """
        if success:
            # Process with feedback flag enabled to reinforce good synonyms
            self.process_text(text, feedback=True)
            
    def save(self) -> None:
        """Save the current state of the learner."""
        self._save_stats()
            

def get_synonym_learner(synonym_library: SynonymLibrary) -> SynonymLearner:
    """
    Factory function to create and initialize a synonym learner.
    
    Args:
        synonym_library: The synonym library to update
        
    Returns:
        Initialized SynonymLearner
    """
    config_manager = get_config_manager()
    
    # Get configuration from config manager
    embedding_model = config_manager.get("default.synonym_learning.embedding_model", "glove-wiki-gigaword-100")
    min_similarity = config_manager.get_float("default.synonym_learning.min_similarity", 0.70)
    confidence_threshold = config_manager.get_float("default.synonym_learning.confidence_threshold", 0.65)
    max_candidates = config_manager.get_int("default.synonym_learning.max_candidates_per_word", 5)
    learning_rate = config_manager.get_float("default.synonym_learning.learning_rate", 0.05)
    use_wordnet = config_manager.get_boolean("default.synonym_learning.use_wordnet", True)
    
    learner = SynonymLearner(
        synonym_library=synonym_library,
        embedding_model=embedding_model,
        min_similarity=min_similarity,
        confidence_threshold=confidence_threshold,
        max_candidates_per_word=max_candidates,
        learning_rate=learning_rate,
        use_wordnet=use_wordnet
    )
    
    return learner
