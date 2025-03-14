"""
Humanizing text processor for UltimateParaphraser.
Provides functions for making text appear more human-written.
"""
import re
import random
import spacy
import nltk
from typing import List, Dict, Tuple, Set, Any, Optional, Literal

from para_humanizer.utils.config import (
    FILLERS, CONNECTORS, CONTRACTIONS, PUNCTUATION_VARIATIONS, 
    INFORMAL_PHRASES, SENTENCE_STRUCTURES
)
from para_humanizer.utils.text_utils import apply_contraction, vary_punctuation

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


class Humanizer:
    """
    Implements methods for adding human-like qualities to text,
    such as fillers, asides, informality, and structural variations.
    """
    
    def __init__(self):
        """Initialize the humanizer processor."""
        self.fillers = FILLERS
        self.connectors = CONNECTORS
        self.contractions = CONTRACTIONS
        self.punctuation_variations = PUNCTUATION_VARIATIONS
        self.informal_phrases = INFORMAL_PHRASES
        self.sentence_structures = SENTENCE_STRUCTURES
        
        # Define formal versions of fillers with less casual language
        self.formal_fillers = [
            "indeed", "certainly", "notably", "significantly",
            "importantly", "evidently", "clearly", "in fact",
            "to clarify", "in particular", "specifically"
        ]
        
        # Define academic hedges that are appropriate for formal writing
        self.academic_hedges = [
            "It appears that ", "Research suggests that ", 
            "Evidence indicates that ", "It may be argued that ",
            "Studies demonstrate that ", "Analysis reveals that ",
            "The data suggests that ", "According to the literature, ",
            "As demonstrated by previous work, ", "Empirical evidence shows that "
        ]
        
        # Define formal thinking patterns for academic writing
        self.formal_thinking_patterns = [
            " more precisely, ", " more specifically, ", 
            " to be precise, ", " to clarify, ", 
            " that is to say, ", " in other words, "
        ]
        
    def add_fillers(self, text: str, intensity: float = 0.5, tone: str = "casual") -> str:
        """
        Add filler words and phrases to make text sound more human.
        
        Args:
            text: The input text
            intensity: Controls how frequently fillers are added (0.0 to 1.0)
            tone: The writing tone ("formal", "academic", "casual")
            
        Returns:
            Text with added filler words
        """
        # For academic writing, drastically reduce or eliminate fillers
        if tone == "academic":
            # Nearly eliminate fillers in academic writing
            if intensity > 0.7:  # Only add minimal formal indicators at high intensity
                rate = 0.03  # Very low rate even with high intensity
                fillers_to_use = self.formal_fillers
            else:
                return text  # No fillers for academic at normal intensities
        elif tone == "formal":
            rate = min(0.08, intensity * 0.16)  # Max 8% for formal writing
            fillers_to_use = self.formal_fillers
        else:  # casual
            # Adjust rate based on intensity
            rate = min(0.15, intensity * 0.3)  # Cap at 15% even with max intensity
            fillers_to_use = self.fillers
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        result = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 4:
                result.append(sentence)
                continue
                
            # Determine if we add a filler to this sentence
            if random.random() < rate:
                words = sentence.split()
                insert_pos = random.randint(1, min(3, len(words) - 1))
                
                # Select a filler and add it appropriately
                filler = random.choice(fillers_to_use)
                
                # Properly format the filler's position
                if insert_pos == 0:
                    # Capitalize first letter if at start
                    if filler[0].islower():
                        filler = filler[0].upper() + filler[1:] + ","
                    words.insert(insert_pos, filler)
                else:
                    # Add commas around filler if in middle
                    words.insert(insert_pos, f", {filler},")
                    
                sentence = ' '.join(words)
                
            result.append(sentence)
            
        return ' '.join(result)
    
    def add_thinking_patterns(self, text: str, intensity: float = 0.5, tone: str = "casual") -> str:
        """
        Add human thinking patterns like self-corrections and second thoughts.
        
        Args:
            text: The input text
            intensity: Controls how frequently patterns are added (0.0 to 1.0)
            tone: The writing tone ("formal", "academic", "casual")
            
        Returns:
            Text with added thinking patterns
        """
        # Academic writing should have very minimal or no informal thinking patterns
        if tone == "academic":
            if intensity > 0.8:  # Only allow minimal formal corrections at very high intensity
                rate = 0.02  # Extremely low rate
                thinking_patterns = self.formal_thinking_patterns
            else:
                return text  # No thinking patterns for academic at normal intensities
        elif tone == "formal":
            rate = min(0.04, intensity * 0.08)  # Max 4% for formal writing
            thinking_patterns = self.formal_thinking_patterns
        else:  # casual
            # Adjust rate based on intensity (relatively rare even at high intensity)
            rate = min(0.08, intensity * 0.16)  # Cap at 8% with max intensity
            thinking_patterns = None  # Use all pattern types for casual
        
        sentences = nltk.sent_tokenize(text)
        result = []
        
        for i, sentence in enumerate(sentences):
            # Skip very short sentences or the last sentence
            if len(sentence.split()) < 6 or i == len(sentences) - 1:
                result.append(sentence)
                continue
                
            # Determine if we add a thinking pattern
            if random.random() < rate:
                # For academic/formal, only use specific formal corrections
                if tone in ["academic", "formal"]:
                    # Find a content word to refine with formal language
                    words = sentence.split()
                    content_words = []
                    
                    for j, word in enumerate(words):
                        # Consider nouns, verbs, adjectives as content words
                        if (len(word) > 3 and word.isalpha() and 
                            j > 0 and j < len(words) - 1):  # Not first or last
                            content_words.append((j, word))
                            
                    if content_words:
                        pos, word = random.choice(content_words)
                        style = random.choice(thinking_patterns)
                        words[pos] = words[pos] + style + word.lower()
                        sentence = ' '.join(words)
                else:
                    # For casual, use the full range of thinking patterns
                    pattern_type = random.randint(0, 3)
                    
                    if pattern_type == 0:  # Self-correction
                        # Find a content word to "correct"
                        words = sentence.split()
                        content_words = []
                        
                        for j, word in enumerate(words):
                            # Consider nouns, verbs, adjectives as content words
                            if (len(word) > 3 and word.isalpha() and 
                                j > 0 and j < len(words) - 1):  # Not first or last
                                content_words.append((j, word))
                                
                        if content_words:
                            pos, word = random.choice(content_words)
                            
                            # Choose a self-correction style
                            correction_styles = [
                                f" - no, I mean {word.lower() if word[0].isupper() else word}",
                                f", or rather {word.lower() if word[0].isupper() else word}",
                                f" (or {word.lower() if word[0].isupper() else word}, actually)"
                            ]
                            
                            style = random.choice(correction_styles)
                            words[pos] = words[pos] + style
                            sentence = ' '.join(words)
                    
                    elif pattern_type == 1:  # Hedge
                        # Add hedging phrase at beginning
                        hedges = [
                            "I think ", "I believe ", "In my opinion, ", 
                            "It seems like ", "From what I understand, ",
                            "As far as I know, ", "If I recall correctly, "
                        ]
                        
                        hedge = random.choice(hedges)
                        
                        # Make sure the first letter after the hedge is lowercase
                        if sentence and sentence[0].isupper():
                            sentence = hedge + sentence[0].lower() + sentence[1:]
                        else:
                            sentence = hedge + sentence
                    
                    elif pattern_type == 2:  # Aside
                        # Add a related aside
                        asides = [
                            " (though I could be wrong)",
                            " (at least that's my understanding)",
                            " (if I'm remembering correctly)",
                            " (based on what I've seen)",
                            " (though there's more to it than that)"
                        ]
                        
                        # Add at end of sentence before period
                        if sentence.endswith(('.', '!', '?')):
                            sentence = sentence[:-1] + random.choice(asides) + sentence[-1]
                        else:
                            sentence = sentence + random.choice(asides)
                    
                    elif pattern_type == 3:  # Second thought
                        # Add a second thought to qualify the statement
                        qualifiers = [
                            " Actually, that's not quite right. ",
                            " On second thought, ",
                            " Wait, let me rephrase that. ",
                            " Actually, to be more precise, ",
                            " Let me clarify that. "
                        ]
                        
                        # Add this to the next sentence
                        next_sentence = sentences[i+1]
                        qualifier = random.choice(qualifiers)
                        
                        # Make sure the first letter after the qualifier is uppercase
                        if next_sentence and next_sentence[0].islower():
                            sentences[i+1] = qualifier + next_sentence[0].upper() + next_sentence[1:]
                        else:
                            sentences[i+1] = qualifier + next_sentence
                    
            result.append(sentence)
            
        return ' '.join(result)
    
    def apply_punctuation_personality(self, text: str, intensity: float = 0.5) -> str:
        """
        Add character to punctuation usage to make it more human.
        
        Args:
            text: The input text
            intensity: Controls how frequently effects are applied (0.0 to 1.0)
            
        Returns:
            Text with more varied punctuation
        """
        # Vary punctuation with a personalized touch
        return vary_punctuation(text, self.punctuation_variations)
    
    def add_conversational_markers(self, text: str, intensity: float = 0.5, tone: str = "casual") -> str:
        """
        Add conversational markers for engaging, personal writing.
        
        Args:
            text: The input text
            intensity: Controls how frequently markers are added (0.0 to 1.0)
            tone: The writing tone ("formal", "academic", "casual")
            
        Returns:
            Text with added conversational elements
        """
        # Adjust rate based on intensity
        if tone == "academic":
            return text  # No conversational markers for academic writing
        elif tone == "formal":
            rate = min(0.06, intensity * 0.12)  # Max 6% for formal writing
        else:  # casual
            rate = min(0.12, intensity * 0.24)  # Cap at 12% with max intensity
        
        sentences = nltk.sent_tokenize(text)
        result = []
        
        for sentence in sentences:
            # Skip very short sentences or questions
            if len(sentence.split()) < 5 or sentence.endswith('?'):
                result.append(sentence)
                continue
                
            # Determine if we add a marker
            if random.random() < rate:
                marker_type = random.randint(0, 3)
                
                if marker_type == 0:  # Add connector at beginning
                    # Add a connector at the beginning of a sentence
                    connector = random.choice(self.connectors)
                    
                    # Check if sentence already starts with a connector
                    first_word = sentence.split()[0].lower()
                    existing_connectors = ['also', 'additionally', 'furthermore', 
                                          'besides', 'however', 'similarly', 
                                          'meanwhile', 'consequently', 'therefore']
                    
                    if first_word not in existing_connectors:
                        sentence = f"{connector}, {sentence[0].lower() + sentence[1:]}"
                
                elif marker_type == 1:  # Informal phrases
                    # Add informal phrase in middle
                    words = sentence.split()
                    
                    if len(words) > 6:
                        insert_pos = random.randint(2, len(words) - 3)
                        informal = random.choice(self.informal_phrases)
                        words.insert(insert_pos, f", {informal},")
                        sentence = ' '.join(words)
                
                elif marker_type == 2:  # Rhetorical questions
                    # Add a rhetorical question at the end
                    questions = [
                        "Right?",
                        "Don't you think?",
                        "Isn't that interesting?",
                        "Wouldn't you agree?",
                        "Makes sense, doesn't it?"
                    ]
                    
                    if sentence.endswith(('.', '!')):
                        sentence = f"{sentence[:-1]} {random.choice(questions)}"
                
                elif marker_type == 3:  # Contractions
                    # Apply contractions with higher probability
                    sentence = apply_contraction(sentence, self.contractions, 0.8)
                    
            result.append(sentence)
            
        return ' '.join(result)
    
    def add_parenthetical_asides(self, text: str, intensity: float = 0.5, tone: str = "casual") -> str:
        """
        Add parenthetical asides for a more personal writing style.
        
        Args:
            text: The input text
            intensity: Controls how frequently asides are added (0.0 to 1.0)
            tone: The writing tone ("formal", "academic", "casual")
            
        Returns:
            Text with added asides
        """
        # Adjust rate based on intensity
        if tone == "academic":
            return text  # No parenthetical asides for academic writing
        elif tone == "formal":
            rate = min(0.04, intensity * 0.08)  # Max 4% for formal writing
        else:  # casual
            rate = min(0.10, intensity * 0.20)  # Cap at 10% with max intensity
        
        sentences = nltk.sent_tokenize(text)
        result = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 6:
                result.append(sentence)
                continue
                
            # Determine if we add an aside
            if random.random() < rate:
                # Parse the sentence for better context
                doc = nlp(sentence)
                
                # Different types of asides
                aside_type = random.randint(0, 3)
                
                if aside_type == 0:  # Personal opinion
                    opinions = [
                        "(which I find fascinating)",
                        "(I think this is important)",
                        "(this really stands out to me)",
                        "(something worth noting)",
                        "(a key point to remember)"
                    ]
                    
                    # Add before period
                    if sentence.endswith(('.', '!', '?')):
                        sentence = sentence[:-1] + " " + random.choice(opinions) + sentence[-1]
                    else:
                        sentence = sentence + " " + random.choice(opinions)
                
                elif aside_type == 1:  # Additional context
                    # Find a noun to add context to
                    nouns = []
                    for token in doc:
                        if token.pos_ == "NOUN" and len(token.text) > 3:
                            nouns.append(token)
                            
                    if nouns:
                        noun = random.choice(nouns)
                        contextual_asides = [
                            f"(a type of {noun.text})",
                            f"(similar to other {noun.text}s)",
                            f"(as {noun.text}s often are)",
                            f"(like many {noun.text}s today)",
                            f"(one of several {noun.text}s)"
                        ]
                        
                        # Get the sentence up to the noun
                        before_noun = sentence[:noun.idx].rstrip()
                        after_noun = sentence[noun.idx + len(noun.text):].lstrip()
                        
                        # Insert the aside after the noun
                        sentence = f"{before_noun} {noun.text} {random.choice(contextual_asides)} {after_noun}"
                
                elif aside_type == 2:  # Clarifying aside
                    clarifications = [
                        "(to be clear)",
                        "(in other words)",
                        "(meaning)",
                        "(that is)",
                        "(specifically)"
                    ]
                    
                    # Add in the middle
                    words = sentence.split()
                    
                    if len(words) > 6:
                        insert_pos = len(words) // 2
                        words.insert(insert_pos, random.choice(clarifications))
                        sentence = ' '.join(words)
                
                elif aside_type == 3:  # Casual aside
                    casual_asides = [
                        "(just saying)",
                        "(believe it or not)",
                        "(crazy, right?)",
                        "(no surprise there)",
                        "(as you might expect)"
                    ]
                    
                    # Add at the end
                    if sentence.endswith(('.', '!', '?')):
                        sentence = sentence[:-1] + " " + random.choice(casual_asides) + sentence[-1]
                    else:
                        sentence = sentence + " " + random.choice(casual_asides)
                        
            result.append(sentence)
            
        return ' '.join(result)
    
    def humanize(self, text: str, intensity: float = 0.5, typo_rate: float = 0.0, tone: str = "casual") -> str:
        """
        Apply all humanization techniques with control over intensity.
        
        Args:
            text: The input text
            intensity: Controls how frequently effects are applied (0.0 to 1.0)
            typo_rate: Rate of introducing typos (0.0 to 1.0)
            tone: The writing tone ("formal", "academic", "casual")
            
        Returns:
            Humanized text
        """
        # Check for empty or invalid text
        if not text or not text.strip():
            return ""
            
        # Apply contractions first for better context
        text = apply_contraction(text, self.contractions, probability=0.7 * intensity)
        
        # Apply each humanization technique with intensity control
        # The order matters for natural results
        
        # 1. Structural changes
        text = self.add_conversational_markers(text, intensity, tone)
        
        # 2. Content additions
        text = self.add_fillers(text, intensity, tone)
        text = self.add_parenthetical_asides(text, intensity, tone)
        
        # 3. Style modifications
        text = self.apply_punctuation_personality(text, intensity)
        
        # 4. Thinking patterns (add last as they can modify structure)
        text = self.add_thinking_patterns(text, intensity, tone)
        
        # 5. If typo rate is set, introduce typos
        if typo_rate > 0:
            words = []
            for word in text.split():
                # Skip very short words and punctuation
                if len(word) <= 3 or not word.isalpha():
                    words.append(word)
                    continue
                    
                # Determine if we should introduce a typo
                if random.random() < typo_rate:
                    # Choose typo type
                    typo_type = random.randint(0, 3)
                    
                    # Preserve capitalization
                    capitalized = word[0].isupper()
                    word_lower = word.lower()
                    
                    if typo_type == 0:  # Character swap
                        if len(word) >= 4:
                            idx = random.randint(1, len(word) - 2)
                            chars = list(word_lower)
                            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                            word = ''.join(chars)
                    
                    elif typo_type == 1:  # Double letter
                        idx = random.randint(0, len(word) - 1)
                        chars = list(word_lower)
                        chars.insert(idx, chars[idx])
                        word = ''.join(chars)
                    
                    elif typo_type == 2:  # Missing letter
                        if len(word) >= 5:  # Only for longer words
                            idx = random.randint(1, len(word) - 2)
                            chars = list(word_lower)
                            chars.pop(idx)
                            word = ''.join(chars)
                    
                    elif typo_type == 3:  # Common adjacent key
                        keyboard_adjacency = {
                            'a': 'sq', 'b': 'vgn', 'c': 'xvd', 'd': 'sf',
                            'e': 'wr', 'f': 'dg', 'g': 'fh', 'h': 'gj',
                            'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
                            'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o',
                            'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry',
                            'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc',
                            'y': 'tu', 'z': 'x'
                        }
                        
                        if len(word) >= 4:
                            idx = random.randint(0, len(word) - 1)
                            if word_lower[idx] in keyboard_adjacency:
                                adjacent_chars = keyboard_adjacency[word_lower[idx]]
                                if adjacent_chars:
                                    replacement = random.choice(adjacent_chars)
                                    chars = list(word_lower)
                                    chars[idx] = replacement
                                    word = ''.join(chars)
                    
                    # Restore capitalization
                    if capitalized:
                        word = word.capitalize()
                
                words.append(word)
            
            text = ' '.join(words)
        
        return text
