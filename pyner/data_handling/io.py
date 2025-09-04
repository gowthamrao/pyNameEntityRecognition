import asyncio
from typing import Any, List, Tuple, Dict, Union, Type, AsyncGenerator, Optional

import pandas as pd
from datasets import Dataset
from pydantic import BaseModel

from pyner.core.engine import CoreEngine
from pyner.models.config import ModelConfig
from pyner.models.factory import ModelFactory
from pyner.schemas.core_schemas import BaseEntity, Entities


def biores_to_entities(tagged_tokens: List[Tuple[str, str]]) -> Entities:
    """
    Converts a list of BIOSES-tagged tokens back into structured entities.
    """
    entities: List[BaseEntity] = []
    current_entity_tokens: List[str] = []
    current_entity_type: Optional[str] = None

    for token, tag in tagged_tokens:
        prefix = tag[0]
        entity_type = tag[2:] if len(tag) > 1 else None

        is_new_entity = prefix in ('B', 'S')
        is_end_of_entity = prefix in ('S', 'E', 'O', 'B')

        if current_entity_tokens and is_end_of_entity:
            if current_entity_type:
                entities.append(BaseEntity(type=current_entity_type, text=" ".join(current_entity_tokens)))
            current_entity_tokens = []
            current_entity_type = None

        if is_new_entity:
            current_entity_tokens.append(token)
            current_entity_type = entity_type
        elif prefix == 'I' and current_entity_type == entity_type:
            current_entity_tokens.append(token)
        elif prefix == 'E' and current_entity_type == entity_type:
            current_entity_tokens.append(token)

    if current_entity_tokens and current_entity_type:
        entities.append(BaseEntity(type=current_entity_type, text=" ".join(current_entity_tokens)))

    return Entities(entities=entities)


async def _yield_texts(input_data: Any, text_column: Optional[str]) -> AsyncGenerator[Tuple[str, Any], None]:
    """Asynchronously yields text and context from various input data types."""
    if isinstance(input_data, str):
        yield input_data, 0
        return

    if isinstance(input_data, list):
        for i, text in enumerate(input_data):
            if isinstance(text, str):
                yield text, i
        return

    if isinstance(input_data, pd.DataFrame):
        if not text_column:
            raise ValueError("`text_column` must be specified for DataFrame inputs.")
        if text_column not in input_data.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        for index, row in input_data.iterrows():
            yield row[text_column], index
        return

    if isinstance(input_data, Dataset):
        if not text_column:
            raise ValueError("`text_column` must be specified for Dataset inputs.")
        if text_column not in input_data.column_names:
            raise ValueError(f"Column '{text_column}' not found in Dataset.")
        for i, item in enumerate(input_data):
            yield item[text_column], i
        return

    raise TypeError(f"Unsupported input data type: {type(input_data)}")


async def extract_entities(
    input_data: Any,
    schema: Type[BaseModel],
    text_column: Optional[str] = None,
    model_config: Optional[Union[Dict, ModelConfig]] = None,
    mode: str = "lcel",
    output_format: str = "conll",
) -> Union[List[Any], Any]:
    """
    High-level public API for extracting entities from various input sources.
    """
    if model_config is None:
        model_config_obj = ModelConfig()
    elif isinstance(model_config, dict):
        model_config_obj = ModelConfig(**model_config)
    else:
        model_config_obj = model_config

    if not issubclass(schema, BaseModel):
        raise TypeError("The provided schema must be a Pydantic BaseModel subclass.")

    model = ModelFactory.create(model_config_obj)
    engine = CoreEngine(model=model, schema=schema)

    texts_to_process = []
    async for text, _ in _yield_texts(input_data, text_column):
        texts_to_process.append(text)

    tasks = [engine.run(text, mode=mode) for text in texts_to_process]
    conll_results = await asyncio.gather(*tasks)

    if output_format == "json":
        output_results = [biores_to_entities(r).model_dump() for r in conll_results]
    elif output_format == "conll":
        output_results = conll_results
    else:
        raise ValueError(f"Unsupported output format: '{output_format}'. Choose 'conll' or 'json'.")

    return output_results[0] if isinstance(input_data, str) else output_results
