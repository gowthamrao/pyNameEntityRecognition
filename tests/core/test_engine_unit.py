import pytest
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.runnables import Runnable

from py_name_entity_recognition.core.engine import CoreEngine

# --- Test Infrastructure: Fake LLM and Schema ---

class UnitTestSchema(BaseModel):
    """A simple schema for unit testing."""
    Person: List[str] = Field(description="The name of a person.")
    Location: List[str] = Field(description="The name of a location.")

class FakeRunnable(Runnable):
    """A fake runnable that returns a pre-defined response."""
    response: Any

    def __init__(self, response: Any):
        self.response = response

    def invoke(self, *args, **kwargs) -> Any:
        return self.response

    async def ainvoke(self, *args, **kwargs) -> Any:
        return self.response

class FakeLLM(BaseLanguageModel):
    """
    A fake LLM for testing. It returns a pre-defined response when its
    structured output is invoked.
    """
    response: Any

    class Config:
        """Pydantic config to allow arbitrary types."""
        arbitrary_types_allowed = True

    def __init__(self, response: Any):
        super().__init__()
        self.response = response

    def with_structured_output(self, schema, **kwargs) -> Runnable:
        """Returns a fake runnable that provides the pre-defined response."""
        return FakeRunnable(self.response)

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs) -> LLMResult:
        return LLMResult(generations=[])

    async def _agenerate(
        self, prompts, stop=None, run_manager=None, **kwargs
    ) -> LLMResult:
        return LLMResult(generations=[])

    def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        raise NotImplementedError()

    def apredict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        raise NotImplementedError()

    def apredict_messages(self, messages, *, stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError()

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        raise NotImplementedError()

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        raise NotImplementedError()

    def predict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        raise NotImplementedError()

    def predict_messages(self, messages, *, stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return "fake-llm-for-testing"

# --- Tests ---

@pytest.mark.asyncio
@pytest.mark.parametrize("text_input, expected_output", [
    ("", []),
    ("   ", []),
    (None, []),
])
async def test_engine_handles_empty_or_invalid_input(text_input, expected_output):
    """Test that the engine gracefully handles empty or whitespace-only input."""
    # Arrange
    fake_llm = FakeLLM(response=None) # The LLM shouldn't even be called.
    engine = CoreEngine(model=fake_llm, schema=UnitTestSchema)

    # Act
    result = await engine.run(text_input, mode="lcel")

    # Assert
    assert result == expected_output


@pytest.mark.asyncio
async def test_engine_filters_llm_hallucinations():
    """
    Test that the engine filters out entities returned by the LLM that are not
    actually present in the source text.
    """
    # Arrange
    text = "Jane Doe works at Acme Corp."

    # The LLM hallucinates a location that is not in the text.
    llm_response = UnitTestSchema(
        Person=["Jane Doe"],
        Location=["Cupertino"] # This is a hallucination.
    )
    fake_llm = FakeLLM(response=llm_response)
    engine = CoreEngine(model=fake_llm, schema=UnitTestSchema)

    # Act
    result = await engine.run(text, mode="lcel")

    # Assert
    # The "Cupertino" entity should be filtered out, and only "Jane Doe" should remain.
    expected = [
        ("Jane", "B-Person"),
        ("Doe", "E-Person"),
        ("works", "O"),
        ("at", "O"),
        ("Acme", "O"),
        ("Corp.", "O"),
    ]
    assert result == expected


@pytest.mark.asyncio
async def test_engine_with_fake_llm_happy_path():
    """Test the engine with a fake LLM in a simple success case."""
    # Arrange
    text = "John Doe lives in New York."

    # The fake LLM will return this Pydantic model instance.
    llm_response = UnitTestSchema(
        Person=["John Doe"],
        Location=["New York"]
    )
    fake_llm = FakeLLM(response=llm_response)

    engine = CoreEngine(model=fake_llm, schema=UnitTestSchema)

    # Act
    result = await engine.run(text, mode="lcel")

    # Assert
    expected = [
        ("John", "B-Person"),
        ("Doe", "E-Person"),
        ("lives", "O"),
        ("in", "O"),
        ("New", "B-Location"),
        ("York", "E-Location"),
        (".", "O"),
    ]

    assert result == expected


@pytest.mark.asyncio
async def test_agentic_mode_self_correction():
    """
    Test that the agentic mode can self-correct a hallucinated entity.
    """
    # Arrange
    text = "Dr. Emily Carter is a chemist."

    # 1. First, the LLM hallucinates a location.
    initial_response = UnitTestSchema(
        Person=["Dr. Emily Carter"],
        Location=["Paris"]  # Hallucination
    )
    # 2. After the refinement prompt, the LLM returns the correct output.
    corrected_response = UnitTestSchema(
        Person=["Dr. Emily Carter"],
        Location=[]
    )

    # The FakeRunnable's ainvoke will be called multiple times.
    # We use a side effect to return different values on each call.
    fake_runnable = FakeRunnable(None)
    fake_runnable.ainvoke = AsyncMock(side_effect=[
        initial_response,
        corrected_response
    ])

    # We need to override the FakeLLM's with_structured_output to return our
    # specially crafted runnable.
    class FakeLLMForAgenticTest(FakeLLM):
        def with_structured_output(self, schema, **kwargs) -> Runnable:
            return fake_runnable

    fake_llm = FakeLLMForAgenticTest(response=None)
    engine = CoreEngine(model=fake_llm, schema=UnitTestSchema, max_retries=1)

    # Act
    result = await engine.run(text, mode="agentic")

    # Assert
    # The final result should not contain the hallucinated location.
    expected = [
        ("Dr.", "B-Person"),
        ("Emily", "I-Person"),
        ("Carter", "E-Person"),
        ("is", "O"),
        ("a", "O"),
        ("chemist", "O"),
        (".", "O"),
    ]
    assert result == expected
    # The LLM should have been called twice (initial + 1 retry).
    assert fake_runnable.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_engine_handles_chunking_and_merging():
    """
    Test that the engine correctly chunks a long text, processes each chunk,
    and merges the results.
    """
    # Arrange
    # A long text that will be split into two chunks by the default chunk size.
    text = ("John Doe, a software engineer from New York, traveled to San Francisco "
            "to meet with Jane Smith, a product manager from Seattle. "
            "They discussed a project at the Google office.")

    # We expect two calls to the LLM, one for each chunk.
    # The FakeLLM needs to return different responses for each call.
    response_chunk_1 = UnitTestSchema(
        Person=["John Doe"],
        Location=["New York", "San Francisco"]
    )
    response_chunk_2 = UnitTestSchema(
        Person=["Jane Smith"],
        Location=["Seattle", "Google office"]
    )

    # We use a side effect on the runnable to simulate different responses
    # for each chunk.
    fake_runnable = FakeRunnable(None)
    fake_runnable.ainvoke = AsyncMock(side_effect=[
        response_chunk_1,
        response_chunk_2
    ])

    class FakeLLMForChunkingTest(FakeLLM):
        def with_structured_output(self, schema, **kwargs) -> Runnable:
            return fake_runnable

    fake_llm = FakeLLMForChunkingTest(response=None)
    # Use a smaller chunk size for easier testing.
    engine = CoreEngine(model=fake_llm, schema=UnitTestSchema, chunk_size=100, chunk_overlap=20)

    # Act
    result = await engine.run(text, mode="lcel")

    # Assert
    # Check that all entities from both chunks are present in the final output.
    result_dict = {token: tag for token, tag in result}

    assert result_dict["John"] == "B-Person"
    assert result_dict["Doe"] == "E-Person"
    assert result_dict["New"] == "B-Location"
    assert result_dict["York"] == "E-Location"
    assert result_dict["San"] == "B-Location"
    assert result_dict["Francisco"] == "E-Location"
    assert result_dict["Jane"] == "B-Person"
    assert result_dict["Smith"] == "E-Person"
    assert result_dict["Seattle"] == "S-Location" # Merged from a single-token entity
    assert result_dict["Google"] == "B-Location"
    assert result_dict["office"] == "E-Location"

    # The LLM should have been called twice (once for each chunk).
    assert fake_runnable.ainvoke.call_count == 2
