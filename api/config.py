"""
Configuration for the Para-Humanizer API.
Contains settings for the API server and security configurations.
"""
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import logging

# Configure logging
logger = logging.getLogger("para-humanizer-api")


class CorsSettings(BaseModel):
    """CORS settings for the API."""
    allow_origins: List[str] = Field(
        default=["*"],
        description="List of allowed origins for CORS"
    )
    allow_credentials: bool = Field(
        default=True,
        description="Whether to allow credentials"
    )
    allow_methods: List[str] = Field(
        default=["*"],
        description="List of allowed HTTP methods"
    )
    allow_headers: List[str] = Field(
        default=["*"],
        description="List of allowed HTTP headers"
    )


class SecuritySettings(BaseModel):
    """Security settings for the API."""
    api_keys_enabled: bool = Field(
        default=False,
        description="Whether API key authentication is enabled"
    )
    api_keys: List[str] = Field(
        default=[],
        description="List of valid API keys"
    )
    rate_limit_enabled: bool = Field(
        default=False,
        description="Whether rate limiting is enabled"
    )
    rate_limit_requests: int = Field(
        default=60,
        description="Maximum number of requests per minute per client"
    )


class ParaphraserSettings(BaseModel):
    """Settings for the paraphraser."""
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU if available"
    )
    batch_size: int = Field(
        default=2,
        description="Batch size for processing"
    )
    hybrid_mode: bool = Field(
        default=True,
        description="Whether to use hybrid mode"
    )
    transformer_disable: bool = Field(
        default=False,
        description="Whether to disable transformer"
    )
    enable_learning: bool = Field(
        default=True,
        description="Whether to enable learning"
    )


class ApiSettings(BaseModel):
    """API settings."""
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    title: str = Field(
        default="Para-Humanizer API",
        description="API title"
    )
    description: str = Field(
        default="Professional API for text paraphrasing with humanization features",
        description="API description"
    )
    debug_mode: bool = Field(
        default=False,
        description="Whether to enable debug mode"
    )


class Settings(BaseModel):
    """Main settings for the application."""
    api: ApiSettings = Field(default_factory=ApiSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    paraphraser: ParaphraserSettings = Field(default_factory=ParaphraserSettings)


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from a config file or environment variables.
    
    Args:
        config_path: Path to a JSON configuration file
        
    Returns:
        Settings object
    """
    settings = Settings()
    
    # If a config file is specified and exists, load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                settings = Settings.model_validate(config_data)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # Check for environment variables (higher priority than config file)
    # API keys from environment
    api_keys_env = os.environ.get("PARA_HUMANIZER_API_KEYS")
    if api_keys_env:
        try:
            settings.security.api_keys = [k.strip() for k in api_keys_env.split(",")]
            settings.security.api_keys_enabled = True
            logger.info("Loaded API keys from environment variables")
        except Exception as e:
            logger.error(f"Error parsing API keys from environment: {e}")
    
    # CORS settings from environment
    cors_origins_env = os.environ.get("PARA_HUMANIZER_CORS_ORIGINS")
    if cors_origins_env:
        try:
            settings.cors.allow_origins = [o.strip() for o in cors_origins_env.split(",")]
            logger.info("Loaded CORS origins from environment variables")
        except Exception as e:
            logger.error(f"Error parsing CORS origins from environment: {e}")
    
    # Rate limiting from environment
    rate_limit_env = os.environ.get("PARA_HUMANIZER_RATE_LIMIT")
    if rate_limit_env and rate_limit_env.isdigit():
        settings.security.rate_limit_requests = int(rate_limit_env)
        settings.security.rate_limit_enabled = True
        logger.info(f"Enabled rate limiting from environment variables: {rate_limit_env} req/min")
    
    # GPU settings from environment
    use_gpu_env = os.environ.get("PARA_HUMANIZER_USE_GPU")
    if use_gpu_env is not None:
        settings.paraphraser.use_gpu = use_gpu_env.lower() in ("true", "1", "yes")
    
    return settings


# Create default settings
settings = load_settings()
