"""
A simple smoke test to ensure that the main components of the package
can be imported, instantiated, and run without crashing.
"""
import pytest
from pyNameEntityRecognition import (
    NERPipeline,
    NERModel,
    ModelConfig,
    PredictionResult,
)

def test_package_imports():
    """Tests that key classes can be imported."""
    # This test passes if the imports above succeed
    assert True

def test_model_instantiation():
    """Tests that the NERModel and its config can be instantiated."""
    config = ModelConfig(
        labels=["PER", "ORG", "LOC"],
        use_crf=True
    )
    model = NERModel(config)
    assert model is not None
    assert model.config.use_crf is True

def test_pipeline_instantiation_and_call():
    """
    Tests that the NERPipeline can be instantiated and called.
    This relies on the placeholder from_pretrained methods.
    """
    # The model name is a dummy for now
    pipeline = NERPipeline.from_pretrained("pyner/bioses-roberta-base")
    assert pipeline is not None

    # Test single prediction
    text = "Acme Corp was founded by John Doe in 1998."
    result = pipeline(text)
    assert isinstance(result, PredictionResult)
    assert result.input_text == text

    # Test batch prediction
    texts = ["First sentence.", "Second sentence."]
    results = pipeline(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], PredictionResult)
    assert results[0].input_text == texts[0]
