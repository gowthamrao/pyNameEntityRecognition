import pytest
import re

from pyner.data_handling.merging import ChunkMerger
from pyner.schemas.core_schemas import BaseEntity


@pytest.fixture
def merger():
    return ChunkMerger()


def test_calculate_confidence_zero_length_chunk(merger):
    confidence = merger._calculate_confidence(0, 0, 0)
    assert confidence == 0.0


def test_merge_invalid_regex(merger, monkeypatch):
    full_text = "He said (hello."
    entities = [BaseEntity(type="GREETING", text="(")]
    chunk_results = [(entities, 0, len(full_text))]

    def mock_finditer(*args, **kwargs):
        raise re.error("test error")

    monkeypatch.setattr("re.finditer", mock_finditer)

    result = merger.merge(full_text, chunk_results)

    # The invalid regex should be skipped, and all tags should be 'O'
    for _, tag in result:
        assert tag == "O"
