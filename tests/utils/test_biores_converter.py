import pytest

from pyner.schemas.core_schemas import BaseEntity
from pyner.utils.biores_converter import BIOSESConverter


@pytest.fixture(scope="module")
def converter():
    """Provides a BIOSESConverter instance for the tests."""
    return BIOSESConverter()


def test_biores_converter_simple_case(converter):
    text = "Apple Inc. is a technology company."
    entities = [BaseEntity(type="ORG", text="Apple Inc.")]
    result = converter.convert(text, entities)

    # Check for presence of tags, regardless of exact tokenization of surrounding text
    assert ("Apple", "B-ORG") in result
    assert ("Inc.", "E-ORG") in result
    # Find the token for 'is' and check it's 'O'
    is_tag = [tag for token, tag in result if token == "is"]
    assert is_tag == ["O"]


def test_biores_converter_single_token_entity(converter):
    text = "I like Google."
    entities = [BaseEntity(type="ORG", text="Google")]
    result = converter.convert(text, entities)

    assert ("Google", "S-ORG") in result
    assert ("like", "O") in result


def test_biores_converter_multiple_entities(converter):
    text = "Apple and Google are rivals."
    entities = [
        BaseEntity(type="ORG", text="Apple"),
        BaseEntity(type="ORG", text="Google"),
    ]
    result = converter.convert(text, entities)

    assert ("Apple", "S-ORG") in result
    assert ("Google", "S-ORG") in result
    assert ("rivals", "O") in result


def test_biores_converter_nested_entities(converter):
    """Tests that longer entities are prioritized over shorter, nested ones."""
    text = "He works at New York University."
    entities = [
        BaseEntity(type="GPE", text="New York"),
        BaseEntity(type="ORG", text="New York University"),
    ]
    result = converter.convert(text, entities)

    assert ("New", "B-ORG") in result
    assert ("York", "I-ORG") in result
    assert ("University", "E-ORG") in result
    # Ensure the shorter entity "New York" was not tagged as GPE
    assert not any(tag.endswith("-GPE") for _, tag in result)


def test_biores_converter_no_entities(converter):
    text = "Just a simple sentence."
    entities = []
    result = converter.convert(text, entities)

    for _, tag in result:
        assert tag == "O"


def test_biores_converter_misaligned_span(converter):
    """Tests that spans not aligning with token boundaries are skipped."""
    text = "The quick brown fox."
    # "ui" is a substring of "quick", not a full token.
    entities = [BaseEntity(type="ANIMAL", text="ui")]
    result = converter.convert(text, entities)

    for _, tag in result:
        assert tag == "O"


def test_biores_converter_three_token_entity(converter):
    text = "I love New York City."
    entities = [BaseEntity(type="GPE", text="New York City")]
    result = converter.convert(text, entities)

    assert ("New", "B-GPE") in result
    assert ("York", "I-GPE") in result
    assert ("City", "E-GPE") in result
