"""
This module defines the Pydantic data structures for the API output.
Using Pydantic models ensures that the output is structured, validated,
and easily serializable.
"""
from pydantic import BaseModel
from typing import List

class TokenDetail(BaseModel):
    """Details of an individual token within an entity."""
    token: str
    start_char: int
    end_char: int
    bioses_tag: str     # The raw BIOSES tag (e.g., "B-PER", "E-PER", "S-LOC")
    token_confidence: float # Confidence score for this specific tag

class Entity(BaseModel):
    """Represents a single recognized named entity."""
    text: str          # The extracted entity text (e.g., "John Doe")
    label: str         # The entity category (e.g., "PER")
    start_char: int
    end_char: int
    confidence: float  # Aggregated confidence score for the entity span
    token_details: List[TokenDetail] # Granular BIOSES details

class PredictionResult(BaseModel):
    """The complete prediction result for a single input text."""
    input_text: str
    entities: List[Entity]
