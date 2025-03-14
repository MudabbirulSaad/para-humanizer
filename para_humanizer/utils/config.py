"""
Configuration module for Para-Humanizer.
Contains default settings and constants used throughout the application.
This module now uses the ConfigManager for all configuration settings.
"""
import os
from typing import Dict, List, Set, Tuple, Any, Optional

from para_humanizer.utils.config_manager import get_config_manager

# Get the configuration manager
config_manager = get_config_manager()

# Default settings
DEFAULT_SETTINGS = config_manager.get_settings()

# Tag mapping for part-of-speech conversion
TAG_MAPPING = config_manager.get_tag_mapping()

# Common filler words for humanizing text
FILLERS = config_manager.get_fillers()

# Sentence connectors for improved flow
CONNECTORS = config_manager.get_connectors()

# Technical terms and phrases that shouldn't be paraphrased
PROTECTED_TERMS = config_manager.get_protected_terms()

# Add specialized academic and scientific terms
ACADEMIC_PROTECTED_TERMS = set([
    # Physics and Quantum Mechanics
    "quantum mechanics", "quantum physics", "quantum theory", "quantum field theory",
    "particle physics", "theoretical physics", "blackbody radiation", "photoelectric effect", 
    "wave-particle duality", "uncertainty principle", "entanglement", "superposition",
    "wavefunction", "quantum state", "quantum computation", "quantum teleportation",
    "quantum cryptography", "quantum information", "double-slit experiment",
    "electron", "electrons", "atom", "atoms", "molecule", "molecules", "particle", "particles",
    "wave", "waves", "orbit", "orbits", "three-dimensional", "bohr model",
    
    # Math and Logic
    "mathematical", "theorem", "axiom", "postulate", "corollary", "lemma", "proof",
    "algorithm", "computation", "formal logic", "set theory", "number theory",
    "calculus", "differential equation", "statistics", "probability", "regression",
    "vector", "matrix", "tensor", "topology", "geometry", "algebra", "analysis",
    
    # Statistics and Research
    "regression analysis", "correlation coefficient", "p value", "statistical significance",
    "coefficient of determination", "r squared", "eigenvalue", "eigenvalues", "variance",
    "dataset", "factor analysis", "component", "components", "variable", "variables",
    "standard deviation", "mean", "median", "mode", "distribution", "normal distribution",
    "statistical", "statistically", "multivariate", "univariate", "anova", "manova",
    "t-test", "chi-square", "confidence interval", "null hypothesis", "alternative hypothesis",
    "sample size", "power analysis", "effect size", "statistical power", "parametric",
    "non-parametric", "dependent variable", "independent variable", "control variable",
    
    # Research Methods
    "methodology", "mixed-methods", "quantitative", "qualitative", "surveys", "interviews",
    "sample", "sampling", "random sampling", "stratified sampling", "cluster sampling",
    "longitudinal", "cross-sectional", "experimental", "quasi-experimental", "control group",
    "treatment group", "blind study", "double-blind", "pilot study", "field study",
    "laboratory experiment", "observational study", "case study", "ethnography",
    "grounded theory", "phenomenology", "content analysis", "discourse analysis",
    
    # Software and Tools
    "spss", "r", "python", "matlab", "sas", "stata", "nvivo", "atlas.ti", "software", "version",
    
    # Philosophy
    "epistemology", "ontology", "metaphysics", "ethics", "aesthetics", "phenomenology",
    "existentialism", "empiricism", "rationalism", "materialism", "idealism", "realism",
    "determinism", "free will", "consciousness", "perception", "cognition",
    
    # Academic Process
    "methodology", "hypothesis", "theory", "experiment", "observation", "empirical",
    "qualitative", "quantitative", "analysis", "synthesis", "conclusion", "abstract",
    "literature review", "case study", "meta-analysis", "correlation", "causation",
    "peer review", "publication", "journal", "citation", "reference", "bibliography",
    "research", "researcher", "researchers", "discourse", "academic", "theoretical", "framework",
    "approach", "evidence", "data", "empirical data", "rigor", "methodological rigor"
])

# Extend protected terms with academic terms
PROTECTED_TERMS.update(ACADEMIC_PROTECTED_TERMS)

# Words to avoid replacing due to common bad replacements
BLACKLIST_WORDS = config_manager.get_blacklist_words()

# Common words that often don't need synonyms
COMMON_WORDS = set([
    "the", "and", "a", "an", "of", "to", "in", "that", "it", "with",
    "for", "on", "is", "was", "be", "as", "are", "were", "am", "been",
    "being", "by", "at", "this", "these", "those", "from", "has", "have",
    "had", "having", "do", "does", "did", "doing", "done", "would", "should",
    "could", "will", "shall", "can", "may", "might", "must", "about", "like",
    "through", "over", "under", "between", "after", "before", "during", "since",
    "until", "while", "because", "though", "although", "if", "unless", "except",
    "but", "yet", "so", "or", "nor", "either", "neither", "both", "whether",
    "not", "only", "just", "very", "too", "quite", "rather", "somewhat", "how",
    "when", "where", "why", "what", "who", "whom", "whose", "which", "there"
])

# Common contractions for more human-like text
CONTRACTIONS = {
    "are not": "aren't",
    "cannot": "can't",
    "could not": "couldn't",
    "did not": "didn't",
    "does not": "doesn't",
    "do not": "don't",
    "had not": "hadn't",
    "has not": "hasn't",
    "have not": "haven't",
    "I am": "I'm",
    "is not": "isn't",
    "it is": "it's",
    "should not": "shouldn't",
    "that is": "that's",
    "they are": "they're",
    "they have": "they've",
    "they will": "they'll",
    "was not": "wasn't",
    "we are": "we're",
    "we have": "we've",
    "we will": "we'll",
    "were not": "weren't",
    "what is": "what's",
    "will not": "won't",
    "would not": "wouldn't",
    "you are": "you're",
    "you have": "you've",
    "you will": "you'll"
}

# Common human error patterns for occasional deliberate errors
COMMON_TYPOS = {
    "the": ["teh", "hte", "th"],
    "and": ["adn", "nad", "an"],
    "that": ["taht", "htat", "tha"],
    "with": ["wiht", "wtih", "wit"],
    "from": ["form", "frmo", "fom"],
    "have": ["hvae", "ahve", "hav"],
    "their": ["thier", "theri", "ther"],
    "would": ["woudl", "wuold", "wold"],
    "about": ["aobut", "abotu", "bout"],
    "there": ["tehre", "thre", "ther"],
    "should": ["shuold", "shoudd", "shoud"],
    "which": ["whcih", "whihc", "wich"],
    "through": ["throught", "throuh", "thru"],
    "could": ["cuold", "coudl", "coud"],
    "because": ["becuase", "becase", "becaus"]
}

# Punctuation variations for more human-like text
PUNCTUATION_VARIATIONS = {
    ".": [".", "...", "!"],
    ",": [",", ";", "—"],
    "?": ["?", "??", "?!"]
}

# Informality markers
INFORMAL_PHRASES = [
    "kind of", "sort of", "like", "you know", "I guess", 
    "pretty much", "more or less", "basically", "literally",
    "I mean", "honestly", "to be fair"
]

# Default transformer model name
DEFAULT_TRANSFORMER_MODEL = "tuner007/pegasus_paraphrase"

# Sentence structure transformations using templates
SENTENCE_STRUCTURES = config_manager.get_sentence_structures()

# Templates for expanding text
EXPANSION_TEMPLATES = config_manager.get_expansion_templates()

# Templates for reducing text
REDUCTION_TEMPLATES = config_manager.get_reduction_templates()

# Humanizing operations and their probabilities
HUMANIZE_OPERATIONS = {
    "add_fillers": 0.35,       # Add filler words
    "vary_contractions": 0.30, # Expand/contract words
    "add_typos": 0.05,         # Add occasional typos
    "add_hedges": 0.15,        # Add hedging language
    "parallel_structure": 0.10,  # Convert to parallel structure
    "passive_to_active": 0.20, # Convert passive to active
    "self_reference": 0.15,    # Add self-reference
    "expand_phrases": 0.25,    # Expand phrases
    "reduce_phrases": 0.15,    # Reduce phrases
    "combine_sentences": 0.10, # Combine adjacent sentences
    "split_sentences": 0.10    # Split complex sentences
}

# Human typo patterns based on keyboard layout and common mistakes
TYPO_PATTERNS = {
    # Common keyboard adjacency errors
    "a": ["s", "q", "z"],
    "b": ["v", "n", "g"],
    "c": ["x", "v", "d"],
    "d": ["s", "f", "e"],
    "e": ["w", "r", "d"],
    "f": ["d", "g", "r"],
    "g": ["f", "h", "t"],
    "h": ["g", "j", "y"],
    "i": ["u", "o", "k"],
    "j": ["h", "k", "u"],
    "k": ["j", "l", "i"],
    "l": ["k", "p", "o"],
    "m": ["n", "j", "k"],
    "n": ["m", "b", "h"],
    "o": ["i", "p", "l"],
    "p": ["o", "l", "["],
    "q": ["w", "a", "s"],
    "r": ["e", "t", "f"],
    "s": ["a", "d", "w"],
    "t": ["r", "y", "g"],
    "u": ["y", "i", "j"],
    "v": ["c", "b", "f"],
    "w": ["q", "e", "s"],
    "x": ["z", "c", "d"],
    "y": ["t", "u", "h"],
    "z": ["a", "x", "s"],
    
    # Common doubling errors
    "double": ["m", "n", "l", "t", "p", "c", "s"],
    
    # Common omission errors
    "omit": ["a", "e", "i", "o", "u", "h", "t", "r", "s"],
    
    # Common transposition patterns
    "transpose": ["on", "in", "er", "re", "es", "se", "th", "ht", "ed", "de", "an", "na"]
}

# The maximum number of threads to use for parallel processing
MAX_WORKERS = max(2, os.cpu_count() or 1)

# Functions to simulate human typing behavior
def get_typo_for_char(char: str) -> str:
    """
    Get a typo for a character based on keyboard layout.
    Used to simulate realistic human typos.
    """
    import random
    if not char.isalpha():
        return char
        
    char = char.lower()
    if char in TYPO_PATTERNS:
        options = TYPO_PATTERNS[char]
        if random.random() < 0.7:  # 70% chance of keyboard adjacency error
            return random.choice(options)
    
    # Keep original character
    return char

class ConfigManager:
    def _load_default_configs(self):
        """Load all default configuration files from the config directory."""
        config_files = [
            'default.json',
            'protected_terms.json',
            'blacklist_words.json',
            'fillers.json',
            'connectors.json',
            'sentence_structures.json',
            'academic_terms.json'  # Add academic terms config
        ]
        # Rest of the function remains the same
