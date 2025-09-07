"""
pyNameEntityRecognition: A Python package for state-of-the-art LLM-based Named Entity Recognition.

This package leverages LangChain for LLM orchestration and LangGraph for creating robust,
agentic, and self-refining extraction workflows. It includes a comprehensive catalog
of predefined schemas for scientific and biomedical text.
"""

__version__ = "0.1.0"

# Expose the primary user-facing function for easy access.
from .data_handling.io import extract_entities

# Expose the catalog features for schema customization and extension.
from .catalog import get_schema, register_entity, PRESETS

__all__ = [
    "extract_entities",
    "get_schema",
    "register_entity",
    "PRESETS",
]
