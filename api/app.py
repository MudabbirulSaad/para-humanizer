"""
FastAPI application for the para-humanizer service.
"""
import time
import uuid
import logging
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Updated middleware import for compatibility with older FastAPI versions
from starlette.middleware.base import BaseHTTPMiddleware
import nltk

from .models import ParaphraseRequest, ParaphraseResponse, ErrorResponse, Usage
from .config import settings
from .security import verify_api_key, get_rate_limit_middleware
from para_humanizer.core.paraphraser import UltimateParaphraser
from para_humanizer.utils.hardware import detect_gpu
from para_humanizer.utils.text_analyzer import text_analyzer

# Configure logging
log_level = logging.DEBUG if settings.api.debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("para-humanizer-api")

# Initialize the paraphraser
paraphraser = UltimateParaphraser(
    use_gpu=settings.paraphraser.use_gpu and detect_gpu(),
    batch_size=settings.paraphraser.batch_size,
    hybrid_mode=settings.paraphraser.hybrid_mode,
    transformer_disable=settings.paraphraser.transformer_disable,
    enable_learning=settings.paraphraser.enable_learning
)

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url="/docs" if not settings.security.api_keys_enabled else None,
    redoc_url="/redoc" if not settings.security.api_keys_enabled else None,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allow_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)

# Add rate limiting middleware if enabled
if settings.security.rate_limit_enabled:
    logger.info(f"Rate limiting enabled: {settings.security.rate_limit_requests} requests per minute")
    app.add_middleware(get_rate_limit_middleware())

# Download required NLTK resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle all exceptions and return proper error responses.
    
    Args:
        request: The request that caused the exception
        exc: The exception raised
        
    Returns:
        JSONResponse with error details
    """
    # Log the error with traceback for debugging
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # In production mode, don't expose internal error details
    if not settings.api.debug_mode:
        error_message = "An internal server error occurred"
    else:
        error_message = str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": error_message, "type": "server_error"}},
    )


@app.get("/", tags=["Health"])
async def health_check():
    """Check if the API is running."""
    return {"status": "ok", "version": settings.api.version}


def count_tokens(text: str) -> int:
    """
    Count tokens in text (simple approximation).
    
    Args:
        text: Text to count tokens in
        
    Returns:
        Approximate token count
    """
    # Simple approximation: 1 token ≈ 4 characters
    return max(1, len(text) // 4)


@app.post("/v1/paraphrase", response_model=ParaphraseResponse, tags=["Paraphrase"])
async def paraphrase(
    request: ParaphraseRequest, 
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Paraphrase the input text with customizable settings.
    
    Args:
        request: Paraphrase request parameters
        api_key: Optional API key for authentication
        
    Returns:
        Paraphrased text with usage statistics
    """
    try:
        start_time = time.time()
        
        # Check if we should determine intelligent parameters
        # Use intelligent parameter selection if all parameters are at default values
        use_intelligent_params = (
            request.rule_based_rate == 0.4 and
            request.transformer_rate == 0.0 and
            request.humanize_intensity == 0.5 and
            request.typo_rate == 0.0
        )
        
        if use_intelligent_params:
            logger.info("Using intelligent parameter selection for paraphrasing")
            params = text_analyzer.analyze(request.text)
            
            # Use the intelligently determined parameters
            rule_based_rate = params['rule_based_rate']
            transformer_rate = params['transformer_rate']
            humanize_intensity = params['humanize_intensity']
            typo_rate = params['typo_rate']
            
            logger.info(f"Intelligent parameters selected: rule_based_rate={rule_based_rate:.2f}, "
                       f"transformer_rate={transformer_rate:.2f}, humanize_intensity={humanize_intensity:.2f}, "
                       f"typo_rate={typo_rate:.4f}")
        else:
            # Use the user-specified parameters
            rule_based_rate = request.rule_based_rate
            transformer_rate = request.transformer_rate
            humanize_intensity = request.humanize_intensity
            typo_rate = request.typo_rate
        
        # Paraphrase the text
        paraphrased_text = paraphraser.paraphrase(
            text=request.text,
            rule_based_rate=rule_based_rate,
            transformer_rate=transformer_rate,
            humanize=request.humanize,
            humanize_intensity=humanize_intensity,
            typo_rate=typo_rate,
            no_parallel=request.no_parallel,
            preserve_structure=request.preserve_structure,
            tone=request.tone
        )
        
        # Calculate token counts (approximation)
        prompt_tokens = count_tokens(request.text)
        completion_tokens = count_tokens(paraphrased_text)
        
        # Prepare and return the response
        response = ParaphraseResponse(
            id=f"paraphrase-{uuid.uuid4()}",
            created=int(time.time()),
            choices=[
                {
                    "index": 0,
                    "text": paraphrased_text,
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        logger.info(f"Paraphrased text with {prompt_tokens} input tokens in {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error paraphrasing text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error paraphrasing text: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=settings.api.debug_mode)
