import pytest
from pydantic import BaseModel, Field
from typing import List, Type, Any
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser

class TestSchema(BaseModel):
    """A Pydantic schema for use in tests."""
    Person: List[str] = Field(description="The name of a person.")
    Location: List[str] = Field(description="A physical location, like a city or country.")

class FakeStructuredLLM(FakeListLLM):
    """
    A fake LLM that simulates the behavior of with_structured_output for testing.
    It inherits from FakeListLLM to provide canned string responses and overrides
    the structured output method to parse that string into a Pydantic model.
    """
    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> Runnable:
        """
        Overrides the base method to return a chain that parses the fake response.
        """
        parser = PydanticOutputParser(pydantic_object=schema)
        # Return a new chain: the fake LLM's response -> parser
        return self | parser

@pytest.fixture(scope="session")
def test_schema() -> Type[BaseModel]:
    """Provides the TestSchema class."""
    return TestSchema

@pytest.fixture
def fake_llm_factory():
    """
    A factory fixture to create FakeStructuredLLM instances.
    This allows tests to simulate specific LLM JSON outputs and test chains
    that rely on the with_structured_output method.
    """
    def _factory(responses: List[str]):
        return FakeStructuredLLM(responses=responses)
    return _factory
