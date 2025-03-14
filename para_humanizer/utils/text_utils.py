"""
Text utility functions for the UltimateParaphraser.
Provides common text processing operations used across the application.
"""
import re
import random
from typing import List, Tuple, Dict, Optional, Set, Any

def fix_formatting(text: str) -> str:
    """
    Fix formatting issues and ensure clean text.
    
    Args:
        text: The input text to clean up
        
    Returns:
        Cleaned text with proper spacing and punctuation
    """
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    
    # Fix spacing after punctuation (but not at end of string)
    text = re.sub(r'([.,;:!?])\s*(?!\Z)', r'\1 ', text)
    
    # Fix spacing around apostrophes
    text = re.sub(r'\s+\'', '\'', text)
    text = re.sub(r'\'\s+', '\'', text)
    
    # Remove double spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Make sure text ends with a single period if needed
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Fix capitalization at start of sentence
    sentences = re.split(r'([.!?])\s+', text)
    fixed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        fixed_sentences.append(sentence + sentences[i+1] + ' ')
        
    if len(sentences) % 2 == 1:
        last_sentence = sentences[-1]
        if last_sentence and last_sentence[0].islower():
            last_sentence = last_sentence[0].upper() + last_sentence[1:]
        fixed_sentences.append(last_sentence)
        
    text = ''.join(fixed_sentences).strip()
    
    return text

def is_protected_term(word: str, text: str, protected_terms: List[str]) -> bool:
    """
    Check if word is part of a protected term that shouldn't be changed.
    
    Args:
        word: The word to check
        text: The full text context
        protected_terms: List of protected terms/phrases
        
    Returns:
        True if the word is part of a protected term
    """
    word = word.lower()
    text = text.lower()
    
    for term in protected_terms:
        if term in text and word in term.split():
            return True
    return False

def apply_contraction(text: str, contractions: Dict[str, str], probability: float = 0.7) -> str:
    """
    Apply contractions more selectively and with better context awareness.
    
    Args:
        text: The input text
        contractions: Dictionary of contractions mapping
        probability: Chance of applying a contraction (0.0 to 1.0)
        
    Returns:
        Text with applied contractions
    """
    # Skip contraction if probability check fails
    if random.random() > probability:
        return text
        
    result = text
    for phrase, contraction in contractions.items():
        # Apply contraction with specified probability for each instance
        # Use lookahead/lookbehind to ensure we're matching word boundaries
        # The negative lookbehind for apostrophes prevents over-contraction
        result = re.sub(
            r'(?<!\w\')(?<!\w-)(?<!\w)\b' + re.escape(phrase) + r'\b(?!\w)',
            lambda m: contraction if random.random() < probability else m.group(0),
            result
        )
        
    return result

def vary_punctuation(text: str, punctuation_variations: Dict[str, List[str]]) -> str:
    """
    Add variety to punctuation usage.
    
    Args:
        text: The input text
        punctuation_variations: Dictionary of punctuation variations
        
    Returns:
        Text with varied punctuation
    """
    result = text
    
    # Replace some punctuation marks with variations
    for punct, variations in punctuation_variations.items():
        # Use lookahead/lookbehind to avoid replacing punctuation in URLs, etc.
        # Only replace some instances (30% chance for each)
        result = re.sub(
            r'(?<!\w)' + re.escape(punct) + r'(?!\w)',
            lambda m: random.choice(variations) if random.random() < 0.3 else m.group(0),
            result
        )
        
    return result

def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of text chunks
    """
    # For small texts, return as single chunk
    if len(text) < max_chunk_size:
        return [text]
        
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into paragraphs first
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # If paragraph is very long, break it into sentences
        if len(paragraph) > max_chunk_size:
            import nltk
            sentences = nltk.sent_tokenize(paragraph)
            
            for sentence in sentences:
                if current_length + len(sentence) > max_chunk_size:
                    # Start a new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            if current_length + len(paragraph) > max_chunk_size:
                # Start a new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def chunk_text_preserve_structure(text: str, max_chunk_size: int = 1000) -> List[Tuple[str, str, int]]:
    """
    Split text into chunks for processing while preserving structure.
    
    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of tuples containing (chunk, structure_type, indentation_level)
        where structure_type can be 'paragraph', 'bullet', 'heading', etc.
    """
    # For small texts, return as single chunk
    if len(text) < max_chunk_size:
        return [(text, 'text', 0)]
    
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        # Preserve empty lines
        if not line.strip():
            if current_chunk:
                # Add current chunk before the empty line
                chunks.append(('\n'.join(current_chunk), 'text', 0))
                current_chunk = []
                current_length = 0
            chunks.append(('', 'empty', 0))
            continue
        
        # Detect line type and indentation
        structure_type = 'text'
        indentation = len(line) - len(line.lstrip())
        
        # Detect bullet points, numbered lists, etc.
        stripped_line = line.strip()
        if re.match(r'^[\*\-•]\s', stripped_line):
            structure_type = 'bullet'
        elif re.match(r'^\d+\.|\d+\)', stripped_line):
            structure_type = 'numbered'
        elif re.match(r'^#+\s', stripped_line):
            structure_type = 'heading'
        
        # If the line is too long, we might need to split it
        if len(line) > max_chunk_size:
            import nltk
            # Try to preserve the indentation and structure marker
            prefix = line[:indentation]
            if structure_type in ('bullet', 'numbered'):
                # Extract the bullet or number
                match = re.match(r'^(\s*)([*\-•]|\d+\.|\d+\))\s', line)
                if match:
                    prefix = match.group(0)
                    indentation = len(prefix)
            
            # Split the content after the prefix
            content = line[indentation:]
            sentences = nltk.sent_tokenize(content)
            
            for i, sentence in enumerate(sentences):
                # Add the prefix only to the first sentence
                if i == 0:
                    sentence = prefix + sentence
                else:
                    sentence = ' ' * indentation + sentence
                
                if current_length + len(sentence) > max_chunk_size and current_chunk:
                    chunks.append(('\n'.join(current_chunk), structure_type, indentation))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            # Normal line processing
            if current_length + len(line) > max_chunk_size and current_chunk:
                chunks.append(('\n'.join(current_chunk), structure_type, indentation))
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line)
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(('\n'.join(current_chunk), 'text', 0))
    
    return chunks

def preserve_structure_paraphrase(text: str, paraphrase_func, **kwargs) -> str:
    """
    Paraphrase text while preserving its original structure.
    
    Args:
        text: The input text
        paraphrase_func: Function to paraphrase individual chunks
        **kwargs: Additional arguments to pass to the paraphrase function
        
    Returns:
        Paraphrased text with preserved structure
    """
    # Extract structural elements (paragraphs, bullet points, etc.)
    chunks = chunk_text_preserve_structure(text)
    result = []
    
    for chunk_text, structure_type, indentation in chunks:
        if not chunk_text.strip():
            # Preserve empty lines exactly
            result.append('')
            continue
            
        # Process the chunk text
        if structure_type in ('text', 'bullet', 'numbered', 'heading'):
            # Paraphrase the content while preserving leading spaces/markers
            if chunk_text:
                # Split into lines to preserve line breaks
                lines = chunk_text.split('\n')
                processed_lines = []
                
                for line in lines:
                    if not line.strip():
                        processed_lines.append(line)
                        continue
                        
                    # Extract leading whitespace and markers
                    leading_space = re.match(r'^\s*', line).group(0)
                    content_match = re.match(r'^(\s*)([*\-•]|\d+\.|\d+\)|\#+)\s', line)
                    
                    if content_match:
                        # Line has a marker (bullet, number, heading)
                        marker = content_match.group(0)
                        content = line[len(marker):]
                        
                        # Paraphrase only the content
                        if content.strip():
                            paraphrased_content = paraphrase_func(content, **kwargs)
                            processed_lines.append(marker + paraphrased_content)
                        else:
                            processed_lines.append(line)
                    else:
                        # Regular line with possible indentation
                        content = line[len(leading_space):]
                        
                        if content.strip():
                            paraphrased_content = paraphrase_func(content, **kwargs)
                            processed_lines.append(leading_space + paraphrased_content)
                        else:
                            processed_lines.append(line)
                
                # Rejoin the lines
                result.append('\n'.join(processed_lines))
            else:
                result.append(chunk_text)
        else:
            # For other structure types or empty lines, preserve as is
            result.append(chunk_text)
    
    # Join all processed chunks
    return '\n'.join(result)
