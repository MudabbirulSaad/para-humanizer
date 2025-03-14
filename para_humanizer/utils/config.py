"""
Configuration module for UltimateParaphraser.
Contains default settings and constants used throughout the application.
"""
import os
from typing import Dict, List, Set, Tuple, Any, Optional

# Default settings
DEFAULT_SETTINGS = {
    "use_gpu": True,
    "batch_size": 8,
    "hybrid_mode": True,
    "transformer_disable": False,
    "rule_based_rate": 0.4,
    "transformer_rate": 0.0,  # Default to 0 for Windows compatibility
    "humanize": True,
    "humanize_intensity": 0.5,
    "typo_rate": 0.005,
    "no_parallel": False
}

# Tag mapping for part-of-speech conversion
TAG_MAPPING = {
    'NN': 'n',
    'NNS': 'n',
    'VB': 'v',
    'VBD': 'v',
    'VBG': 'v',
    'VBN': 'v',
    'JJ': 'a',
    'JJR': 'a',
    'JJS': 'a',
    'RB': 'r',
    'RBR': 'r',
    'RBS': 'r'
}

# Common filler words for humanizing text
FILLERS = [
    "actually", "basically", "honestly", "simply", 
    "essentially", "really", "definitely", "probably",
    "personally", "frankly", "seriously", "clearly",
    "obviously", "of course", "you know", "I mean",
    "to be fair", "to be honest", "in my opinion", "well"
]

# Sentence connectors for improved flow
CONNECTORS = [
    "Also", "Additionally", "Furthermore", "Besides",
    "On the other hand", "However", "Similarly",
    "In addition", "Meanwhile", "Consequently",
    "As a result", "Therefore", "For instance",
    "In fact", "In contrast", "Interestingly"
]

# Technical terms and phrases that shouldn't be paraphrased
PROTECTED_TERMS = set([
    'covid-19', 'covid', 'sars-cov-2', 'coronavirus',
    'python', 'javascript', 'java', 'c++', 'react', 'angular', 'vue', 'node.js',
    'microsoft', 'google', 'apple', 'amazon', 'facebook', 'twitter', 'linkedin',
    'bitcoin', 'ethereum', 'blockchain', 'cryptocurrency', 'crypto',
    'machine learning', 'artificial intelligence', 'ai', 'ml', 'deep learning',
    'neural network', 'transformer', 'attention', 'nlp', 'natural language processing',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
    'cpu', 'gpu', 'tpu', 'ram', 'ssd', 'hdd', 'storage', 'memory', 'processor',
    'html', 'css', 'xml', 'json', 'yaml', 'api', 'rest', 'graphql', 'http', 'https',
    'sql', 'nosql', 'database', 'mysql', 'postgresql', 'mongodb', 'redis',
    'aws', 'azure', 'gcp', 'cloud', 'serverless', 'docker', 'kubernetes', 'container',
    'git', 'github', 'gitlab', 'bitbucket', 'version control', 'ci/cd', 'devops',
    'linux', 'windows', 'macos', 'ios', 'android', 'operating system', 'os',
    'ultimateparaphraser',  # Add our tool name as protected
])

# Words to avoid replacing due to common bad replacements
BLACKLIST_WORDS = set([
    # Inappropriate or potentially offensive words
    'ass', 'arse', 'bitch', 'bastard', 'crap', 'cunt', 'damn', 'dick', 'douchebag',
    'fag', 'faggot', 'fuck', 'fucked', 'fucking', 'goddamn', 'hell', 'jackass', 'jerk',
    'piss', 'pissed', 'prick', 'pussy', 'shit', 'slut', 'twat', 'wanker', 'whore',
    'cock', 'penis', 'dick',
    
    # Professional-sounding but still potentially inappropriate
    'idiotic', 'stupid', 'dumb', 'moronic', 'lame', 'crazy', 'insane', 'nuts',
    'retarded', 'crippled', 'handicapped', 'spastic', 'autistic', 'schizo',
    
    # Words that often lead to confusing replacements
    'virtually', 'literally', 'actually', 'basically', 'essentially', 'utterly',
    'utter', 'complete', 'absolutely', 'totally', 'utter', 'sheer', 'pure',
    'perfect', 'entire', 'pure', 'mere', 'outright', 'downright', 'thorough',
    
    # Ambiguous or technical terms that make poor substitutions
    'technical', 'systematic', 'inherent', 'intrinsic', 'ostensible', 'putative',
    'cognizant', 'salient', 'nominal', 'purported', 'substantive', 'normative',
    
    # Additional inappropriate terms
    'anal', 'anus', 'aroused', 'ballsack', 'blowjob', 'boner', 'boob', 'breast',
    'butt', 'butthole', 'climax', 'cum', 'ejaculate', 'erection', 'fetish',
    'foreskin', 'genital', 'hardcore', 'horny', 'intercourse', 'jizz', 'labia',
    'masturbate', 'nipple', 'nude', 'oral', 'orgasm', 'porn', 'pornography',
    'rectal', 'rectum', 'scrotum', 'semen', 'sex', 'sexual', 'sexy', 'skank',
    'sperm', 'testicle', 'tit', 'vagina', 'vulva', 'wank',
    
    # Additional terms that could be inappropriate in professional context
    'cheap', 'trashy', 'nasty', 'crappy', 'lousy', 'junky', 'shoddy', 'worthless',
    'pathetic', 'useless', 'incompetent', 'inept', 'ridiculous', 'absurd',
    
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "this", "that", "these", "those",
    "and", "but", "or", "for", "with", "by", "about", "it",
    "artificial", "intelligence", "machine", "learning",
    "not", "no", "none", "never", "ever", "always", "sometimes",
    "often", "seldom", "rarely", "frequently", "occasionally",
    "every", "each", "any", "some", "all", "both", "either", "neither",
    "as", "so", "very", "too", "quite", "rather", "extremely",
    "few", "many", "much", "more", "most", "less", "least",
    "over", "under", "before", "after", "during", "until", "while",
    "if", "unless", "although", "though", "because", "since",
    "they", "them", "their", "theirs", "themselves",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "i", "me", "my", "mine", "myself",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how"
])

# Common words that don't need synonyms as often
COMMON_WORDS = set([
    "should", "would", "could", "may", "might", "must", "shall",
    "day", "week", "month", "year", "time", "place", "person",
    "case", "fact", "point", "world", "life", "man", "woman", 
    "child", "home", "room", "area", "money", "job", "school",
    "university", "company", "business", "government"
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

# Sentence structure templates for variety
SENTENCE_STRUCTURES = [
    # Inversion templates
    ("It is {adj} that {subject} {verb}", "{subject} {verb}, which is {adj}"),
    ("There are {plural_noun} that {plural_verb}", "{plural_noun} {plural_verb}"),
    ("It is necessary to {verb}", "We need to {verb}"),
    ("This is a {noun} that {verb}", "This {noun} {verb}"),
    
    # Voice change templates
    ("{subject} {verb} {object}", "{object} {passive_verb} by {subject}"),
    ("{subject} {verb} that", "It is {past_participle} by {subject} that"),
    
    # Conjunction templates
    ("{clause1}. {clause2}", "{clause1}, and {clause2}"),
    ("{clause1}. {clause2}", "{clause1}, but {clause2}"),
    ("{clause1}. {clause2}", "{clause1} while {clause2}"),
    ("{clause1}. {clause2}", "Although {clause1}, {clause2}"),
    ("{clause1}. {clause2}", "Since {clause1}, {clause2}"),
    ("{clause1}. {clause2}", "{clause1}, which {clause2}")
]

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

# The maximum number of threads to use for parallel processing
MAX_WORKERS = max(2, os.cpu_count() or 1)
