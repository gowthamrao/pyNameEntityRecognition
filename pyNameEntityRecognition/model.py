"""
This module defines the core model architecture (NERModel) and its
configuration object (ModelConfig).
"""
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from pydantic import BaseModel
from typing import List, Dict

# Using Pydantic for config ensures type safety and easy serialization
class ModelConfig(BaseModel):
    """
    Configuration for an NERModel.
    """
    encoder_backbone: str = "bert-base-cased"
    labels: List[str]
    use_crf: bool = True # Not used yet, but in the config

class NERModel(nn.Module):
    """
    The underlying PyTorch module for NER.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Create BIOSES label mappings
        bioses_labels = ["O"]
        for label in self.config.labels:
            bioses_labels.extend([f"B-{label}", f"I-{label}", f"S-{label}", f"E-{label}"])

        self.id2label: Dict[int, str] = {i: label for i, label in enumerate(bioses_labels)}
        self.label2id: Dict[str, int] = {label: i for i, label in enumerate(bioses_labels)}

        # 2. Load the transformer backbone
        # We also grab the encoder's config to get hidden size, etc.
        encoder_config = AutoConfig.from_pretrained(config.encoder_backbone)
        self.encoder = AutoModel.from_pretrained(config.encoder_backbone)

        # 3. Define the classification head
        # Use a default for hidden_dropout_prob if not present in the config
        dropout_prob = getattr(encoder_config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(encoder_config.hidden_size, len(self.id2label))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass for the NER model.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """
        Loads a pre-trained model architecture and weights.
        (This is a placeholder implementation).
        The smoke test instantiates via NERModel(config), so we keep that working.
        """
        # This part requires a more complete implementation of saving/loading
        # which is out of scope for this step. We'll leave it as a placeholder
        # but a more informative one.
        raise NotImplementedError(
            "Loading a fully pre-trained NERModel is not implemented yet. "
            "Instantiate with `NERModel(config=...)` to create a new model from a backbone."
        )

    def save_pretrained(self, save_directory: str):
        """
        Saves the model and its configuration to a directory.
        (This is a placeholder implementation).
        """
        print(f"Saving model to {save_directory}...")
        # Here we would save weights and the config.json
        print("Model saved (placeholder).")
