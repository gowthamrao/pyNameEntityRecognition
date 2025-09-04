import pytest
from unittest.mock import patch
from pydantic import ValidationError

from pyner.models.config import ModelConfig
from pyner.models.factory import ModelFactory


@patch('pyner.models.factory.ChatOpenAI')
def test_create_openai_model(mock_chat_openai, mocker):
    """Tests the creation of a ChatOpenAI model."""
    mocker.patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    config = ModelConfig(provider="openai", model_name="gpt-4o")
    ModelFactory.create(config)
    mock_chat_openai.assert_called_once()

@patch('pyner.models.factory.ChatAnthropic')
def test_create_anthropic_model(mock_chat_anthropic, mocker):
    """Tests the creation of a ChatAnthropic model."""
    mocker.patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    config = ModelConfig(provider="anthropic", model_name="claude-3-opus")
    ModelFactory.create(config)
    mock_chat_anthropic.assert_called_once()

@patch('pyner.models.factory.ChatOllama')
def test_create_ollama_model(mock_chat_ollama):
    """Tests the creation of a ChatOllama model."""
    config = ModelConfig(provider="ollama", model_name="llama3")
    ModelFactory.create(config)
    mock_chat_ollama.assert_called_once()

@patch('pyner.models.factory.AzureChatOpenAI')
def test_create_azure_model(mock_azure_chat_openai, mocker):
    """Tests the creation of an AzureChatOpenAI model."""
    mocker.patch.dict('os.environ', {
        'AZURE_OPENAI_API_KEY': 'test_key',
        'OPENAI_API_VERSION': '2023-05-15'
    })
    config = ModelConfig(
        provider="azure",
        azure_deployment="my-deployment",
        azure_endpoint="https://my-endpoint.openai.azure.com/",
        api_version="2023-05-15"
    )
    ModelFactory.create(config)
    mock_azure_chat_openai.assert_called_once()

def test_unsupported_provider_config_raises_error():
    """Tests that an unsupported provider raises a ValidationError on ModelConfig."""
    with pytest.raises(ValidationError):
        ModelConfig(provider="unsupported_provider")

def test_factory_unsupported_provider_raises_error(mocker):
    """Tests that the factory raises a ValueError for an unsupported provider."""
    config = ModelConfig()
    mocker.patch.object(config, 'provider', 'unsupported_provider')
    with pytest.raises(ValueError, match="Unsupported model provider: 'unsupported_provider'"):
        ModelFactory.create(config)
