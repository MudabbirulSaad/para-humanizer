#!/usr/bin/env python
"""
Run the Para-Humanizer API server.
"""
import logging
import argparse
import uvicorn
import os

from api.config import load_settings, Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("para-humanizer-api")


def main():
    """Run the FastAPI server for Para-Humanizer."""
    parser = argparse.ArgumentParser(description="Para-Humanizer API Server")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        help="Comma-separated list of API keys"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        help="Rate limit in requests per minute"
    )
    args = parser.parse_args()

    # Load configuration
    settings = load_settings(args.config)
    
    # Override settings with command line arguments
    if args.debug:
        settings.api.debug_mode = True
        os.environ["PARA_HUMANIZER_DEBUG"] = "true"
        
    if args.api_keys:
        settings.security.api_keys = [k.strip() for k in args.api_keys.split(",")]
        settings.security.api_keys_enabled = len(settings.security.api_keys) > 0
        os.environ["PARA_HUMANIZER_API_KEYS"] = args.api_keys
        
    if args.rate_limit:
        settings.security.rate_limit_requests = args.rate_limit
        settings.security.rate_limit_enabled = True
        os.environ["PARA_HUMANIZER_RATE_LIMIT"] = str(args.rate_limit)
    
    # Log API configuration
    auth_status = "enabled" if settings.security.api_keys_enabled else "disabled"
    rate_limit_status = "enabled" if settings.security.rate_limit_enabled else "disabled"
    debug_status = "enabled" if settings.api.debug_mode else "disabled"
    
    logger.info(f"Starting Para-Humanizer API on {args.host}:{args.port}")
    logger.info(f"API authentication: {auth_status}")
    logger.info(f"Rate limiting: {rate_limit_status}")
    logger.info(f"Debug mode: {debug_status}")
    
    # Run the server
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload or settings.api.debug_mode,
    )


if __name__ == "__main__":
    main()
