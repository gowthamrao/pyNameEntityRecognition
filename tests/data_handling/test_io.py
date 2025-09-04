import pytest
import pandas as pd
from datasets import Dataset
from pydantic import BaseModel
from unittest.mock import patch, MagicMock, AsyncMock

from pyner.data_handling.io import (
    biores_to_entities,
    _yield_texts,
    extract_entities,
)
from pyner.schemas.core_schemas import BaseEntity, Entities

class Person(BaseEntity):
    pass

class Location(BaseEntity):
    pass

@pytest.mark.parametrize("tagged_tokens, expected_entities", [
    ([("John", "S-PERSON"), ("works", "O"), ("at", "O"), ("Google", "S-ORG")],
     Entities(entities=[BaseEntity(type='PERSON', text='John'), BaseEntity(type='ORG', text='Google')])),
    ([("New", "B-LOCATION"), ("York", "E-LOCATION")],
     Entities(entities=[BaseEntity(type='LOCATION', text='New York')])),
    ([("San", "B-LOCATION"), ("Francisco", "I-LOCATION"), ("Bay", "E-LOCATION")],
     Entities(entities=[BaseEntity(type='LOCATION', text='San Francisco Bay')])),
    ([], Entities(entities=[])),
])
def test_biores_to_entities(tagged_tokens, expected_entities):
    assert biores_to_entities(tagged_tokens) == expected_entities

@pytest.mark.asyncio
async def test_yield_texts():
    # Test with string
    assert [x async for x in _yield_texts("hello", None)] == [("hello", 0)]

    # Test with list
    assert [x async for x in _yield_texts(["a", "b"], None)] == [("a", 0), ("b", 1)]

    # Test with DataFrame
    df = pd.DataFrame({"text": ["c", "d"], "other": [1, 2]})
    assert [x async for x in _yield_texts(df, "text")] == [("c", 0), ("d", 1)]

    # Test with Dataset
    ds = Dataset.from_pandas(df)
    assert [x async for x in _yield_texts(ds, "text")] == [("c", 0), ("d", 1)]

    # Test unsupported type
    with pytest.raises(TypeError):
        [x async for x in _yield_texts(123, None)]

    # Test missing text_column
    with pytest.raises(ValueError):
        [x async for x in _yield_texts(df, None)]

@pytest.mark.asyncio
@patch('pyner.data_handling.io.CoreEngine')
@patch('pyner.data_handling.io.ModelFactory')
async def test_extract_entities(mock_factory, mock_engine):
    mock_engine_instance = mock_engine.return_value
    mock_engine_instance.run = AsyncMock(return_value=[("John", "S-PERSON")])

    # Test with string
    result = await extract_entities("John", Person, output_format="json")
    assert result == {'entities': [{'type': 'PERSON', 'text': 'John'}]}

    # Test with list
    results = await extract_entities(["John", "Doe"], Person, output_format="conll")
    assert len(results) == 2
    assert results[0] == [("John", "S-PERSON")]

    # Test with invalid schema
    with pytest.raises(TypeError):
        await extract_entities("test", "not a schema")

    # Test with invalid output format
    with pytest.raises(ValueError):
        await extract_entities("test", Person, output_format="invalid")
