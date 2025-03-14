"""
Configuration Manager for Para-Humanizer.
Provides unified access to all configuration settings through a centralized API.
"""
import os
import json
import logging
from typing import Dict, List, Set, Any, Optional, Union
import copy
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration management system for Para-Humanizer.
    Handles loading configuration from JSON files and environment variables,
    providing a unified API for accessing all configuration settings.
    """
    
    def __init__(self, resources_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            resources_dir: Path to the resources directory. If None, uses the default path.
        """
        if resources_dir is None:
            # Use the default resources directory
            self.resources_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "resources"
            )
        else:
            self.resources_dir = resources_dir
            
        self.config_dir = os.path.join(self.resources_dir, "config")
        self._config_cache = {}
        self._loaded_configs = set()
        
        # Load the default configuration first
        self.load_config("default")
        
    def load_config(self, config_name: str) -> bool:
        """
        Load a configuration file into memory.
        
        Args:
            config_name: Name of the configuration file (without .json extension)
            
        Returns:
            True if successful, False otherwise
        """
        if config_name in self._loaded_configs:
            return True
            
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        try:
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return False
                
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Strip meta information
            if "meta" in data:
                data.pop("meta")
                
            # Store the configuration
            self._config_cache[config_name] = data
            self._loaded_configs.add(config_name)
            
            logger.info(f"Loaded configuration: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration {config_name}: {str(e)}")
            return False
            
    def reload_config(self, config_name: str) -> bool:
        """
        Reload a configuration file from disk.
        
        Args:
            config_name: Name of the configuration file (without .json extension)
            
        Returns:
            True if successful, False otherwise
        """
        if config_name in self._loaded_configs:
            self._loaded_configs.remove(config_name)
            
        if config_name in self._config_cache:
            del self._config_cache[config_name]
            
        return self.load_config(config_name)
        
    def reload_all_configs(self) -> bool:
        """
        Reload all loaded configuration files from disk.
        
        Returns:
            True if all reloads were successful, False otherwise
        """
        configs_to_reload = list(self._loaded_configs)
        self._loaded_configs.clear()
        self._config_cache.clear()
        
        all_successful = True
        for config_name in configs_to_reload:
            success = self.load_config(config_name)
            all_successful = all_successful and success
            
        return all_successful
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        Keys can be hierarchical using dot notation (e.g., 'settings.use_gpu').
        
        Args:
            key: Configuration key to retrieve
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        # Check environment variables first (takes precedence)
        env_var = f"PARA_HUMANIZER_{key.upper().replace('.', '_')}"
        env_value = os.environ.get(env_var)
        if env_value is not None:
            # Convert environment variable to appropriate type
            if env_value.lower() in ("true", "false"):
                return env_value.lower() == "true"
            try:
                if "." in env_value:
                    return float(env_value)
                return int(env_value)
            except ValueError:
                return env_value
        
        # Try to find the key in loaded configurations
        parts = key.split(".")
        if len(parts) == 1:
            # This is a top-level key, look for it in all loaded configs
            for config_name in self._loaded_configs:
                if parts[0] in self._config_cache[config_name]:
                    return self._config_cache[config_name][parts[0]]
        else:
            # This is a hierarchical key, look for it in the specified config
            config_name = parts[0]
            if config_name in self._loaded_configs:
                current = self._config_cache[config_name]
                for part in parts[1:]:
                    if part in current:
                        current = current[part]
                    else:
                        return default
                return current
        
        return default
        
    def get_list(self, key: str) -> List[Any]:
        """
        Get a configuration list by key.
        
        Args:
            key: Configuration key to retrieve
            
        Returns:
            Configuration list or empty list if not found
        """
        value = self.get(key, [])
        if not isinstance(value, list):
            # Try to convert to list if it's a string
            if isinstance(value, str):
                return [item.strip() for item in value.split(",")]
            return [value]
        return value
        
    def get_dict(self, key: str) -> Dict[str, Any]:
        """
        Get a configuration dictionary by key.
        
        Args:
            key: Configuration key to retrieve
            
        Returns:
            Configuration dictionary or empty dict if not found
        """
        value = self.get(key, {})
        if not isinstance(value, dict):
            return {}
        return value
        
    def get_set(self, key: str) -> Set[str]:
        """
        Get a configuration set by key.
        
        Args:
            key: Configuration key to retrieve
            
        Returns:
            Configuration set or empty set if not found
        """
        value = self.get_list(key)
        return set(value)
        
    def get_boolean(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value to return if key not found
            
        Returns:
            Boolean configuration value or default
        """
        value = self.get(key, default)
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "on")
        return bool(value)
        
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get an integer configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value to return if key not found
            
        Returns:
            Integer configuration value or default
        """
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
            
    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Get a float configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value to return if key not found
            
        Returns:
            Float configuration value or default
        """
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
            
    def set(self, key: str, value: Any, config_name: str = "default") -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key to set
            value: Value to set
            config_name: Name of the configuration to modify
            
        Returns:
            True if successful, False otherwise
        """
        if config_name not in self._loaded_configs:
            if not self.load_config(config_name):
                return False
                
        parts = key.split(".")
        if len(parts) == 1:
            self._config_cache[config_name][parts[0]] = value
        else:
            # Handle hierarchical keys
            current = self._config_cache[config_name]
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
            
        return True
        
    def save_config(self, config_name: str) -> bool:
        """
        Save a configuration to disk.
        
        Args:
            config_name: Name of the configuration to save
            
        Returns:
            True if successful, False otherwise
        """
        if config_name not in self._loaded_configs:
            logger.error(f"Cannot save unloaded configuration: {config_name}")
            return False
            
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        try:
            # Load existing file to preserve meta information
            meta = {"version": "1.0.0", "date_updated": "2025-03-14"}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if "meta" in existing_data:
                        meta = existing_data["meta"]
                        
            # Create a new dictionary with meta and configuration data
            data_to_save = {
                "meta": meta,
                **self._config_cache[config_name]
            }
            
            # Write the configuration to disk
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved configuration: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration {config_name}: {str(e)}")
            return False
            
    def get_protected_terms(self) -> Set[str]:
        """
        Get the set of protected terms that shouldn't be paraphrased.
        
        Returns:
            Set of protected terms
        """
        self.load_config("protected_terms")
        return self.get_set("protected_terms.protected_terms")
        
    def get_blacklist_words(self) -> Set[str]:
        """
        Get the set of blacklisted words that should never be used as synonyms.
        
        Returns:
            Set of blacklisted words
        """
        self.load_config("blacklist_words")
        return self.get_set("blacklist_words.blacklist_words")
        
    def get_fillers(self) -> List[str]:
        """
        Get the list of filler words for humanizing text.
        
        Returns:
            List of filler words
        """
        self.load_config("fillers")
        return self.get_list("fillers.fillers")
        
    def get_connectors(self) -> List[str]:
        """
        Get the list of sentence connectors for improving text flow.
        
        Returns:
            List of connectors
        """
        self.load_config("connectors")
        return self.get_list("connectors.connectors")
        
    def get_sentence_structures(self) -> List[List[str]]:
        """
        Get the list of sentence structure templates for rule-based transformations.
        
        Returns:
            List of sentence structure template pairs (from, to)
        """
        self.load_config("sentence_structures")
        return self.get_list("sentence_structures.sentence_structures")
        
    def get_expansion_templates(self) -> List[List[str]]:
        """
        Get the list of expansion templates for rule-based transformations.
        
        Returns:
            List of expansion template pairs
        """
        self.load_config("sentence_structures")
        return self.get_list("sentence_structures.expansion_templates")
        
    def get_reduction_templates(self) -> List[List[str]]:
        """
        Get the list of reduction templates for rule-based transformations.
        
        Returns:
            List of reduction template pairs
        """
        self.load_config("sentence_structures")
        return self.get_list("sentence_structures.reduction_templates")
        
    def get_tag_mapping(self) -> Dict[str, str]:
        """
        Get the tag mapping for part-of-speech conversion.
        
        Returns:
            Dictionary mapping POS tags to simplified tags
        """
        self.load_config("default")
        return self.get_dict("default.tag_mapping")
        
    def get_settings(self) -> Dict[str, Any]:
        """
        Get the default settings for the paraphraser.
        
        Returns:
            Dictionary of default settings
        """
        self.load_config("default")
        return self.get_dict("default.settings")


# Singleton instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Get the singleton instance of the ConfigManager.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
