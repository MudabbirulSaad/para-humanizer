"""
API models for the paraphraser service.
"""
from typing import List, Optional, Dict, Any, TypeVar
from pydantic import BaseModel, Field, field_validator, model_validator


T = TypeVar('T', bound='ParaphraseRequest')

class ParaphraseRequest(BaseModel):
    """
    Model for a paraphrase request, following OpenAI-like API conventions.
    """
    text: str = Field(..., description="The input text to paraphrase")
    rule_based_rate: float = Field(0.4, ge=0.0, le=1.0, 
                                  description="The rate of rule-based word replacement (0.0 to 1.0)")
    transformer_rate: float = Field(0.0, ge=0.0, le=1.0,
                                   description="The rate of transformer-based paraphrasing (0.0 to 1.0)")
    humanize: bool = Field(True, description="Whether to apply humanization techniques")
    humanize_intensity: float = Field(0.5, ge=0.0, le=1.0, 
                                     description="The intensity of humanization (0.0 to 1.0)")
    typo_rate: float = Field(0.0, ge=0.0, le=1.0,
                            description="The rate of introducing typos (0.0 to 1.0)")
    no_parallel: bool = Field(False, description="Whether to disable parallel processing")
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    @model_validator(mode='after')
    def check_rates(self) -> 'ParaphraseRequest':
        if self.rule_based_rate == 0.0 and self.transformer_rate == 0.0:
            raise ValueError("At least one of rule_based_rate or transformer_rate must be greater than 0")
        return self


class Usage(BaseModel):
    """
    Model for tracking API usage statistics.
    """
    prompt_tokens: int = Field(..., description="Number of tokens in the input text")
    completion_tokens: int = Field(..., description="Number of tokens in the paraphrased text")
    total_tokens: int = Field(..., description="Total number of tokens processed")


class ParaphraseResponse(BaseModel):
    """
    Model for a paraphrase response, following OpenAI-like API conventions.
    """
    id: str = Field(..., description="Unique identifier for the request")
    object: str = Field("paraphrase", description="Object type")
    created: int = Field(..., description="Unix timestamp of when the request was created")
    model: str = Field("ultimate-paraphraser-v1", description="Model used for paraphrasing")
    choices: List[Dict[str, Any]] = Field(..., description="Array of paraphrased results")
    usage: Usage = Field(..., description="Usage statistics for the request")


class ErrorResponse(BaseModel):
    """
    Model for API error responses.
    """
    error: Dict[str, Any] = Field(..., description="Error details")
