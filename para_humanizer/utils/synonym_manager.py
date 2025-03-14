"""
Synonym Manager for Para-Humanizer.
Command-line utility for managing the synonym library.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# Ensure we can import from our package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from para_humanizer.utils.config_manager import get_config_manager
from para_humanizer.utils.synonym_loader import SynonymLibrary, DEFAULT_SYNONYM_PATH

def load_synonym_library(filepath: Optional[str] = None) -> Tuple[SynonymLibrary, str]:
    """Load the synonym library from the given filepath or default location."""
    config_manager = get_config_manager()
    blacklist_words = config_manager.get_blacklist_words()
    common_words = config_manager.get_set("default.common_words")
    
    library = SynonymLibrary(blacklist_words, common_words)
    path = filepath or DEFAULT_SYNONYM_PATH
    
    if not library.load_synonyms(path):
        print(f"Error: Could not load synonyms from {path}")
        sys.exit(1)
        
    return library, path

def add_synonym(args):
    """Add a synonym to the library."""
    library, path = load_synonym_library(args.filepath)
    config_manager = get_config_manager()
    blacklist_words = config_manager.get_blacklist_words()
    
    # Convert words to lowercase for consistency
    word = args.word.lower()
    synonym = args.synonym.lower()
    
    # Check if the word or synonym is in the blacklist
    if word in blacklist_words:
        print(f"Error: The word '{word}' is in the blacklist and cannot be added.")
        return
        
    if synonym in blacklist_words:
        print(f"Error: The synonym '{synonym}' is in the blacklist and cannot be added.")
        return
    
    if library.add_synonym(word, synonym, args.category):
        print(f"Added '{synonym}' as a synonym for '{word}' in category '{args.category or 'auto-detected'}'")
        library.save_synonyms(path)
    else:
        print(f"Could not add synonym. The synonym may already exist or the library might not be loaded properly.")

def remove_synonym(args):
    """Remove a synonym from the library."""
    library, path = load_synonym_library(args.filepath)
    
    # Convert words to lowercase for consistency
    word = args.word.lower()
    synonym = args.synonym.lower() if args.synonym else None
    
    removed = False
    
    # If no specific synonym is provided, remove all synonyms for the word
    if not synonym:
        for category in library.synonyms:
            if word in library.synonyms[category]:
                del library.synonyms[category][word]
                if word in library.word_to_category:
                    del library.word_to_category[word]
                removed = True
                print(f"Removed all synonyms for '{word}' from category '{category}'")
    else:
        # Remove specific synonym
        if word in library.word_to_category:
            category = library.word_to_category[word]
            if word in library.synonyms[category]:
                if synonym in library.synonyms[category][word]:
                    library.synonyms[category][word].remove(synonym)
                    removed = True
                    print(f"Removed '{synonym}' as a synonym for '{word}'")
                    
                    # If no synonyms left, remove the word
                    if not library.synonyms[category][word]:
                        del library.synonyms[category][word]
                        del library.word_to_category[word]
                        print(f"No synonyms left for '{word}', removed from library")
    
    if removed:
        library.save_synonyms(path)
    else:
        print(f"Could not find the specified word or synonym in the library.")

def list_synonyms(args):
    """List synonyms in the library."""
    library, _ = load_synonym_library(args.filepath)
    
    if args.word:
        word = args.word.lower()
        found = False
        
        # Look for the word in all categories
        for category, words in library.synonyms.items():
            if word in words:
                print(f"Synonyms for '{word}' in category '{category}':")
                for synonym in words[word]:
                    print(f"  - {synonym}")
                found = True
                
        if not found:
            print(f"No synonyms found for '{word}'")
    elif args.category:
        category = args.category
        if category in library.synonyms:
            print(f"Words in category '{category}':")
            
            # Sort words alphabetically for better readability
            for word in sorted(library.synonyms[category].keys()):
                print(f"  - {word}: {', '.join(library.synonyms[category][word])}")
        else:
            print(f"Category '{category}' not found in the library")
    else:
        # List all categories and word counts
        print("Synonym Library Summary:")
        for category in sorted(library.synonyms.keys()):
            word_count = len(library.synonyms[category])
            synonym_count = sum(len(synonyms) for synonyms in library.synonyms[category].values())
            print(f"  - {category}: {word_count} words, {synonym_count} synonyms")
            
        # Overall stats
        total_words = len(library.word_to_category)
        total_synonyms = sum(len(synonyms) for category in library.synonyms.values() 
                            for synonyms in category.values())
        print(f"\nTotal: {total_words} words with {total_synonyms} synonyms across {len(library.synonyms)} categories")

def export_library(args):
    """Export the library to a JSON file."""
    library, _ = load_synonym_library(args.filepath)
    
    output_path = args.output_file
    
    if library.save_synonyms(output_path):
        print(f"Successfully exported synonym library to {output_path}")
    else:
        print(f"Failed to export synonym library")

def import_library(args):
    """Import synonyms from a JSON file."""
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        return
        
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Load the current library
        library, path = load_synonym_library(args.filepath)
        
        # Skip meta information
        if 'meta' in data:
            data.pop('meta')
            
        word_count = 0
        synonym_count = 0
        
        # Import each category
        for category, words in data.items():
            if category not in library.synonyms:
                library.synonyms[category] = {}
                
            for word, synonyms in words.items():
                word = word.lower()
                
                # Skip blacklisted words
                config_manager = get_config_manager()
                blacklist_words = config_manager.get_blacklist_words()
                if word in blacklist_words:
                    continue
                    
                if word not in library.synonyms[category]:
                    library.synonyms[category][word] = []
                    library.word_to_category[word] = category
                    word_count += 1
                    
                # Add new synonyms
                for synonym in synonyms:
                    synonym = synonym.lower()
                    if synonym not in blacklist_words and synonym not in library.synonyms[category][word]:
                        library.synonyms[category][word].append(synonym)
                        synonym_count += 1
        
        # Save the updated library
        if library.save_synonyms(path):
            print(f"Successfully imported {word_count} words with {synonym_count} synonyms from {args.input_file}")
        else:
            print(f"Failed to save the updated synonym library")
            
    except Exception as e:
        print(f"Error importing synonyms: {str(e)}")

def create_new_library(args):
    """Create a new empty synonym library."""
    output_path = args.output_file
    
    # Create basic structure
    data = {
        "meta": {
            "version": "1.0.0",
            "description": "Curated synonym dictionary for Para-Humanizer",
            "date_updated": "2025-03-14"
        },
        "nouns": {},
        "verbs": {},
        "adjectives": {},
        "adverbs": {},
        "technical_terms": {}
    }
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print(f"Created new empty synonym library at {output_path}")
        
    except Exception as e:
        print(f"Error creating new library: {str(e)}")

def main():
    """Main entry point for the synonym manager."""
    parser = argparse.ArgumentParser(description="Manage the Para-Humanizer synonym library")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add synonym command
    add_parser = subparsers.add_parser("add", help="Add a synonym to the library")
    add_parser.add_argument("word", help="Word to add a synonym for")
    add_parser.add_argument("synonym", help="Synonym to add")
    add_parser.add_argument("--category", help="Category for the word (e.g., technical_terms, verbs)")
    add_parser.add_argument("--filepath", help="Path to synonym library JSON file")
    add_parser.set_defaults(func=add_synonym)
    
    # Remove synonym command
    remove_parser = subparsers.add_parser("remove", help="Remove a synonym from the library")
    remove_parser.add_argument("word", help="Word to remove a synonym from")
    remove_parser.add_argument("synonym", help="Synonym to remove")
    remove_parser.add_argument("--filepath", help="Path to synonym library JSON file")
    remove_parser.set_defaults(func=remove_synonym)
    
    # List synonyms command
    list_parser = subparsers.add_parser("list", help="List synonyms for a word")
    list_parser.add_argument("word", nargs="?", help="Word to list synonyms for (optional)")
    list_parser.add_argument("--category", help="Only list words from this category")
    list_parser.add_argument("--filepath", help="Path to synonym library JSON file")
    list_parser.set_defaults(func=list_synonyms)
    
    # Export library command
    export_parser = subparsers.add_parser("export", help="Export the synonym library to a JSON file")
    export_parser.add_argument("output_file", help="Path to output JSON file")
    export_parser.add_argument("--filepath", help="Path to source synonym library JSON file")
    export_parser.set_defaults(func=export_library)
    
    # Import library command
    import_parser = subparsers.add_parser("import", help="Import synonyms from a JSON file")
    import_parser.add_argument("input_file", help="Path to input JSON file")
    import_parser.add_argument("--filepath", help="Path to target synonym library JSON file")
    import_parser.add_argument("--replace", action="store_true", help="Replace existing library instead of merging")
    import_parser.set_defaults(func=import_library)
    
    # Create new library command
    create_parser = subparsers.add_parser("create", help="Create a new empty synonym library")
    create_parser.add_argument("output_file", help="Path to output JSON file")
    create_parser.set_defaults(func=create_new_library)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    args.func(args)

if __name__ == "__main__":
    main()
