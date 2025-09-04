"""
A simple smoke test to ensure that the main components of the package
can be imported, instantiated, and run without crashing.
"""
import pytest
import torch
from transformers import AutoTokenizer
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
    """
    # Use a real (but small) model from the hub and provide dummy labels
    model_name = "distilbert-base-uncased"
    labels = ["PER", "ORG", "LOC"]
    pipeline = NERPipeline.from_pretrained(model_name, labels=labels)
    assert pipeline is not None
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None

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

def test_model_forward_pass():
    """Tests the forward pass of the NERModel and checks output shape."""
    # 1. Config and Model setup
    model_name = "distilbert-base-uncased"
    labels = ["PER", "ORG"]
    config = ModelConfig(
        encoder_backbone=model_name,
        labels=labels,
    )
    model = NERModel(config)

    # 2. Create dummy input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt")

    # 3. Forward pass
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # 4. Check output shape
    # Shape should be (batch_size, sequence_length, num_labels)
    num_bioses_labels = 1 + len(labels) * 4 # O + (B, I, S, E for each label)
    assert logits.shape == (1, inputs["input_ids"].shape[1], num_bioses_labels)
