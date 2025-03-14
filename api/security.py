"""
Security components for the Para-Humanizer API.
Includes API key authentication and rate limiting.
"""
import time
from typing import Dict, Optional, Callable
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import logging

from .config import settings

# Configure logging
logger = logging.getLogger("para-humanizer-api")

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting storage
class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    Attributes:
        max_requests: Maximum number of requests per minute
    """
    
    def __init__(self, max_requests: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests per minute
        """
        self.max_requests = max_requests
        self.requests = defaultdict(list)
    
    def is_rate_limited(self, client_id: str) -> bool:
        """
        Check if a client is rate limited.
        
        Args:
            client_id: Client identifier (IP address)
            
        Returns:
            True if rate limited, False otherwise
        """
        # Get current time
        current_time = time.time()
        
        # Remove requests older than 60 seconds
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < 60
        ]
        
        # Check if rate limited
        if len(self.requests[client_id]) >= self.max_requests:
            return True
        
        # Add current request
        self.requests[client_id].append(current_time)
        
        return False


# Create rate limiter instance
rate_limiter = RateLimiter(settings.security.rate_limit_requests)


async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> Optional[str]:
    """
    Verify the API key if authentication is enabled.
    
    Args:
        api_key: API key from the request header
        
    Returns:
        API key if valid, None if authentication is disabled
        
    Raises:
        HTTPException: If the API key is invalid
    """
    # Skip validation if API keys are not enabled
    if not settings.security.api_keys_enabled:
        return None
        
    # Validate that an API key was provided
    if not api_key:
        logger.warning("Missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
        
    if api_key not in settings.security.api_keys:
        logger.warning(f"Invalid API key used: {api_key[:5]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
        
    return api_key


# Rate limiting middleware using Starlette's BaseHTTPMiddleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Middleware to implement rate limiting.
        
        Args:
            request: The incoming request
            call_next: The next middleware in the chain
            
        Returns:
            The response from the next middleware
            
        Raises:
            HTTPException: If the rate limit is exceeded
        """
        if not settings.security.rate_limit_enabled:
            return await call_next(request)
        
        # Get client IP as identifier
        client_id = request.client.host if request.client else "unknown"
        
        # Check if rate limited
        if rate_limiter.is_rate_limited(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {settings.security.rate_limit_requests} "
                      f"requests per minute allowed."
            )
        
        # Process the request
        return await call_next(request)


# Function to get the middleware for use in app.py
def get_rate_limit_middleware():
    return RateLimitMiddleware
