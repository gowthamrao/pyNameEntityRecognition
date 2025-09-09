# pyNameEntityRecognition

**pyNameEntityRecognition** is a state-of-the-art Python package for LLM-based Named Entity Recognition (NER). It leverages LangChain for LLM orchestration and LangGraph for creating robust, agentic, and self-refining extraction workflows. The package is designed to be highly flexible, provider-agnostic, and easily extensible.

## Key Features

- **Dual-Path Extraction Engine**: Choose between a high-throughput, streamlined LCEL pipeline for efficiency (`mode='lcel'`) or a stateful, self-correcting agentic workflow using LangGraph for maximum accuracy (`mode='agentic'`).
- **Dynamic Schema Definition**: Define your entity extraction targets on the fly using simple Pydantic models. The engine automatically adapts its prompts and parsing to your schema.
- **LLM Agnostic**: Seamlessly switch between different LLM providers like OpenAI, Anthropic, Azure, and local models via Ollama. The engine accepts any pre-instantiated LangChain `BaseLanguageModel` object.
- **Automatic Chunking & Merging**: For documents that exceed the LLM's context window, the engine automatically handles NLP-aware text chunking and intelligently merges the results from overlapping chunks.
- **Anti-Hallucination**: The agentic workflow includes a critical validation step that ensures every extracted entity span is an exact, verbatim match of the source text, significantly reducing hallucinations.
- **Developer-Friendly**: Comes with built-in observability features, including structured logging, seamless LangSmith integration, and a `displaCy`-like visualizer for NER outputs.

## Installation

1.  **Install the package:**
    Clone the repository and install the package using pip. It is recommended to do this in a virtual environment.

    ```bash
    pip install .
    ```

2.  **Install spaCy Model:**
    The package uses spaCy for robust tokenization. You'll need to download the default English model.

    ```bash
    python -m spacy download en_core_web_sm
    ```

3.  **Install Development Dependencies (for testing):**
    If you wish to run the test suite, you will need to install the development dependencies manually (due to a limitation in some `pip` versions with `pyproject.toml` dependency groups).

    ```bash
    pip install pytest pytest-asyncio pytest-mock
    ```

## Configuration

### LLM API Keys

The package uses `python-dotenv` to automatically load environment variables from a `.env` file in your project's root directory. Create a `.env` file to store your API keys:

```
# .env file
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="..."
```

### LangSmith Observability (Recommended)

To enable detailed tracing and debugging with [LangSmith](https://www.langchain.com/langsmith), add the following variables to your `.env` file:

```
# .env file
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="Your-Project-Name" # Optional: The project to log runs to
```

## Quick Start

Here's a simple example of how to use the main `extract_entities` function.

```python
import asyncio
from pydantic import BaseModel, Field
from typing import List

# Import the main extraction function
from py_name_entity_recognition import extract_entities
# Import the visualization utility
from py_name_entity_recognition.observability.visualization import display_biores

# 1. Define your extraction schema using Pydantic
class UserInfo(BaseModel):
    """Schema for extracting user information."""
    Person: List[str] = Field(description="The full name of a person.")
    Location: List[str] = Field(description="A city, state, or country.")
    Company: List[str] = Field(description="The name of a company or organization.")

# 2. Define the text you want to process
text = "John Doe, a software engineer at Google, lives in New York. He is meeting with Jane Smith from Microsoft tomorrow."

# 3. Run the extraction
async def main():
    # Use the default 'lcel' mode for fast extraction
    print("--- Running in LCEL Mode ---")
    conll_output = await extract_entities(
        input_data=text,
        schema=UserInfo
    )
    print(conll_output)

    # Display the result as color-coded HTML (works best in Jupyter)
    display_biores(conll_output)

    # Get the output in a structured JSON format
    print("\n--- Running with JSON Output ---")
    json_output = await extract_entities(
        input_data=text,
        schema=UserInfo,
        output_format="json"
    )
    print(json_output)

    # Use the 'agentic' mode for higher accuracy and self-correction
    print("\n--- Running in Agentic Mode ---")
    agentic_output = await extract_entities(
        input_data=text,
        schema=UserInfo,
        mode="agentic"
    )
    display_biores(agentic_output)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates the core functionality of the package, including schema definition, running different extraction modes, and formatting the output.
