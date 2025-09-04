import pytest
import json
from pyner.core.engine import CoreEngine

pytestmark = pytest.mark.asyncio

@pytest.fixture
def sample_text():
    return "Alice lives in Paris and Bob lives in London."

async def test_engine_lcel_path_success(fake_llm_factory, test_schema, sample_text):
    """Tests the end-to-end LCEL path with a successful extraction."""
    responses = [
        json.dumps({
            "Person": ["Alice", "Bob"],
            "Location": ["Paris", "London"]
        })
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run(sample_text, mode="lcel")

    result_dict = {token: tag for token, tag in result}
    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Bob"] == "S-Person"
    assert result_dict["Paris"] == "S-Location"
    assert result_dict["London"] == "S-Location"
    assert result_dict["lives"] == "O"

async def test_engine_agentic_path_success(fake_llm_factory, test_schema, sample_text):
    """Tests the agentic path where validation succeeds on the first try."""
    responses = [
        json.dumps({
            "Person": ["Alice", "Bob"],
            "Location": ["Paris", "London"]
        })
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run(sample_text, mode="agentic")

    result_dict = {token: tag for token, tag in result}
    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Bob"] == "S-Person"

async def test_engine_agentic_path_refinement(fake_llm_factory, test_schema, sample_text):
    """Tests the agentic path with one round of refinement."""
    responses = [
        json.dumps({"Person": ["Alice"], "Location": ["Zurich"]}), # Bad
        json.dumps({"Person": ["Alice", "Bob"], "Location": ["Paris", "London"]}) # Good
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema, max_retries=1)

    result = await engine.run(sample_text, mode="agentic")

    result_dict = {token: tag for token, tag in result}
    assert result_dict["London"] == "S-Location"
    assert "Zurich" not in result_dict

async def test_engine_agentic_path_max_retries(fake_llm_factory, test_schema, sample_text):
    """Tests that the agentic path gives up after max retries."""
    bad_response = json.dumps({"Person": [], "Location": ["Zurich"]})
    responses = [bad_response] * 4 # Initial call + 3 retries
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema, max_retries=3)

    result = await engine.run(sample_text, mode="agentic")

    # Should fail gracefully and return all 'O' tags
    for _, tag in result:
        assert tag == "O"

async def test_engine_chunking_and_merging(fake_llm_factory, test_schema):
    """Tests the chunking and merging logic for long documents."""
    long_text = "Alice is a doctor in New York. " * 50

    responses = [
        json.dumps({"Person": ["Alice"], "Location": ["New York", "Old York"]}),
        json.dumps({"Person": ["Alice"], "Location": ["New York"]}),
    ]
    llm = fake_llm_factory(responses * 10)
    engine = CoreEngine(model=llm, schema=test_schema, chunk_size=100, chunk_overlap=20)

    result = await engine.run(long_text, mode="lcel")

    found_alice = any(token == "Alice" and tag == "S-Person" for token, tag in result)
    found_ny = any(token == "New" and tag == "B-Location" for token, tag in result)
    found_old_york = any("Old" in token for token, tag in result)

    assert found_alice
    assert found_ny
    assert not found_old_york
