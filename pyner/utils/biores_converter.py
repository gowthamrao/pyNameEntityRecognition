import re
from typing import List, Tuple

import spacy
from spacy.language import Language

from pyner.observability.logging import logger
from pyner.schemas.core_schemas import BaseEntity


class BIOSESConverter:
    """
    Converts structured entity predictions into BIOSES-tagged tokens.

    This class uses a spaCy model to tokenize the source text and then maps the
    character-level spans of extracted entities to the token level, assigning
    the appropriate BIOSES (Beginning, Inside, Outside, Single, End) tag.
    """

    def __init__(self, nlp: Language = None):
        """
        Initializes the converter with a spaCy Language object.

        Args:
            nlp: A loaded spaCy Language object. If None, it will attempt to load
                 'en_core_web_sm' by default.

        Raises:
            IOError: If the default spaCy model cannot be loaded.
        """
        if nlp:
            self.nlp = nlp
        else:
            logger.info("No spaCy model provided. Loading 'en_core_web_sm' by default.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error(
                    "Could not load 'en_core_web_sm'. Please run the following command to install it:\n"
                    "python -m spacy download en_core_web_sm"
                )
                raise IOError("Default spaCy model not found.")

    def convert(self, text: str, entities: List[BaseEntity]) -> List[Tuple[str, str]]:
        """
        Converts a list of structured entities into BIOSES-tagged tokens.

        The method handles nested entities by prioritizing longer spans and ensures
        that entity spans align perfectly with token boundaries.

        Args:
            text: The original source text from which entities were extracted.
            entities: A list of BaseEntity objects representing the extracted spans.

        Returns:
            A list of (token, tag) tuples, representing the tokenized text with
            BIOSES tags. For example: [('John', 'S-PERSON'), ('Doe', 'S-PERSON')].
        """
        doc = self.nlp(text)
        tags = ["O"] * len(doc)

        # Handle cases where no entities are provided.
        if not entities:
            return list(zip([token.text for token in doc], tags))

        # Sort entities by length (descending) to handle nested cases correctly.
        # Larger spans (e.g., "New York City") are processed before smaller,
        # nested ones (e.g., "New York").
        sorted_entities = sorted(entities, key=lambda e: len(e.text), reverse=True)

        for entity in sorted_entities:
            try:
                # Escape special regex characters in the entity text to ensure literal matching.
                pattern = re.escape(entity.text)
                for match in re.finditer(pattern, text):
                    start_char, end_char = match.span()

                    # Use spaCy's `doc.char_span` for precise, token-aligned spans.
                    # This is crucial for ensuring the extracted text corresponds to
                    # a valid token sequence.
                    span = doc.char_span(start_char, end_char, label=entity.type)

                    if span is None:
                        logger.warning(
                            f"Entity span '{entity.text}' at chars ({start_char}-{end_char}) "
                            "does not align with token boundaries. Skipping this occurrence."
                        )
                        continue

                    # Check if any token in the span has already been tagged.
                    # Since we sorted by length, this prevents shorter, nested entities
                    # from overwriting the tags of longer, containing entities.
                    if any(tags[i] != "O" for i in range(span.start, span.end)):
                        logger.debug(
                            f"Skipping entity '{entity.text}' due to token overlap with an "
                            "already-tagged (and longer) entity."
                        )
                        continue

                    # Apply the BIOSES tags based on the number of tokens in the span.
                    if len(span) == 1:
                        tags[span.start] = f"S-{entity.type}"
                    else:
                        tags[span.start] = f"B-{entity.type}"
                        for i in range(span.start + 1, span.end - 1):
                            tags[i] = f"I-{entity.type}"
                        tags[span.end - 1] = f"E-{entity.type}"

            except re.error as e:
                logger.warning(f"Could not compile regex for entity text: '{entity.text}'. Error: {e}")
                continue

        return list(zip([token.text for token in doc], tags))
