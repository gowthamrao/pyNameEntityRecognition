"""
pyNameEntityRecognition: A Python package for state-of-the-art LLM-based Named Entity Recognition.

This package leverages LangChain for LLM orchestration and LangGraph for creating robust,
agentic, and self-refining extraction workflows.
"""

__version__ = "0.1.0"

# Expose the primary user-facing function for easy access.
from .data_handling.io import extract_entities

__all__ = ["extract_entities"]
