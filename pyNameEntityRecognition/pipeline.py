"""
This module contains the NERPipeline, the primary entry point for end-users
to perform inference.
"""
import torch
from typing import List, Union, Optional
from .schemas import PredictionResult
from .model import NERModel, ModelConfig
from transformers import AutoTokenizer

class NERPipeline:
    """
    The main inference pipeline for Named Entity Recognition.
    """
    def __init__(self, model: NERModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, labels: Optional[List[str]] = None):
        """
        Loads a pre-trained model and tokenizer from the Hugging Face Hub
        or a local directory.

        This method can be used in two ways:
        1. Load a fully pre-trained NER model (if the directory contains the
           necessary config with labels). This is not yet fully supported.
        2. Initialize a new NER model for fine-tuning from a pre-trained
           transformer backbone. In this case, the `labels` argument is required.
        """
        # For now, we only support initializing from a backbone, which requires labels.
        if labels is None:
            # In the future, we would try to load a saved ModelConfig here.
            raise ValueError(
                "When initializing from a transformer backbone, the `labels` argument is required."
            )

        # 1. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # 2. Create model config and instantiate model
        config = ModelConfig(encoder_backbone=model_name_or_path, labels=labels)
        model = NERModel(config)

        return cls(model, tokenizer)

    def __call__(self, inputs: Union[str, List[str]], batch_size: int = 32, device: str = "auto") -> Union[PredictionResult, List[PredictionResult]]:
        """
        Performs NER on a single text or a batch of texts.
        """
        # 1. Handle single string input
        is_single_input = isinstance(inputs, str)
        if is_single_input:
            inputs = [inputs]

        # 2. Device management
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        results = []
        # 3. Batch processing
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]

            # 4. Tokenization
            # We need offset_mapping to map tokens back to original characters
            tokenized_inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
                max_length=512, # A sensible default
            )

            # The tokenizer gives us offset mappings, which we need for post-processing
            # We pop it from the dict before sending the rest to the model.
            offset_mapping = tokenized_inputs.pop("offset_mapping")

            # Move tensors to the correct device
            tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

            # 5. Inference
            with torch.no_grad():
                logits = self.model(**tokenized_inputs)

            # 6. Decoding (simple argmax)
            predictions = torch.argmax(logits, dim=-1)

            # 7. Post-processing (convert IDs to entities)
            # This is where the complex logic to group B-I-E-S tags and handle
            # subwords would go. For now, we'll just return the empty structure.
            for text in batch_inputs:
                results.append(PredictionResult(input_text=text, entities=[]))

        return results[0] if is_single_input else results
