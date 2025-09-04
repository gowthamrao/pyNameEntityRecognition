# This file makes pyNameEntityRecognition a Python package.

# Expose key classes for easy access
from .schemas import TokenDetail, Entity, PredictionResult
from .pipeline import NERPipeline
from .model import NERModel, ModelConfig
