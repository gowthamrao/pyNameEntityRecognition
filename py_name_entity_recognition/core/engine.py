import asyncio
import json
from typing import List, Optional, Tuple, Type, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from py_name_entity_recognition.data_handling.chunking import chunk_text_with_offsets
from py_name_entity_recognition.data_handling.merging import ChunkMerger
from py_name_entity_recognition.prompting.prompt_manager import PromptManager, ZeroShotStructured
from py_name_entity_recognition.schemas.core_schemas import BaseEntity
from py_name_entity_recognition.utils.biores_converter import BIOSESConverter


class AgenticGraphState(TypedDict, total=False):
    original_text: str
    extraction_schema: Type[BaseModel]
    llm_output: Optional[BaseModel]
    validated_entities: Optional[List[BaseEntity]]
    validation_errors: Optional[List[str]]
    retry_count: int


class CoreEngine:
    def __init__(
        self,
        model: BaseLanguageModel,
        schema: Type[BaseModel],
        max_retries: int = 3,
        chunk_size: int = 2000,
        chunk_overlap: int = 300,
    ):
        self.model = model
        self.schema = schema
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.prompt_manager = PromptManager(strategy=ZeroShotStructured())
        self.biores_converter = BIOSESConverter()
        self.merger = ChunkMerger()
        self._agentic_graph_app = self._build_agentic_graph().compile()

    async def run(self, text: str, mode: str = "lcel") -> List[Tuple[str, str]]:
        if len(text) <= self.chunk_size:
            entities = await self._get_intermediate_entities(text, mode)
            return self.biores_converter.convert(text, entities)

        chunks_with_offsets = chunk_text_with_offsets(
            text, self.chunk_size, self.chunk_overlap
        )
        tasks = [
            self._get_intermediate_entities(chunk, mode)
            for chunk, _ in chunks_with_offsets
        ]
        chunk_entities_list = await asyncio.gather(*tasks)
        chunk_results = [
            (entities, offset, offset + len(chunk))
            for (chunk, offset), entities in zip(
                chunks_with_offsets, chunk_entities_list
            )
        ]
        return self.merger.merge(text, chunk_results)

    async def _get_intermediate_entities(
        self, text: str, mode: str
    ) -> List[BaseEntity]:
        if mode == "lcel":
            llm_output = await self._run_lcel_chain(text)
            return self._transform_to_base_entities(llm_output)
        elif mode == "agentic":
            final_state = await self._agentic_graph_app.ainvoke(
                {
                    "original_text": text,
                    "extraction_schema": self.schema,
                    "retry_count": 0,
                }
            )
            return final_state.get("validated_entities") or []
        else:
            raise ValueError(f"Unknown execution mode: '{mode}'")

    async def _run_lcel_chain(self, text_input: str) -> BaseModel:
        prompt_template = self.prompt_manager.get_prompt_template(self.schema)
        structured_llm = self.model.with_structured_output(
            self.schema.model_json_schema()
        )
        chain = prompt_template | structured_llm
        result = await chain.ainvoke({"text_input": text_input})
        return self.schema.model_validate(result)

    def _build_agentic_graph(self) -> StateGraph:
        graph = StateGraph(AgenticGraphState)
        graph.add_node("extract", self._extract_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("refine", self._refine_node)
        graph.add_conditional_edges(
            "validate",
            self._decide_to_refine_or_end,
            {"refine": "refine", END: END},
        )
        graph.set_entry_point("extract")
        graph.add_edge("extract", "validate")
        # After refining, we re-validate the new output.
        graph.add_edge("refine", "validate")
        return graph

    async def _extract_node(self, state: AgenticGraphState) -> dict:
        llm_output = await self._run_lcel_chain(state["original_text"])
        return {"llm_output": llm_output, "validation_errors": None}

    def _validate_node(self, state: AgenticGraphState) -> dict:
        llm_output = state.get("llm_output")
        original_text = state["original_text"]
        errors = []
        if not llm_output:
            errors.append("LLM output is empty or invalid.")
            return {"validation_errors": errors, "validated_entities": None}
        entities = self._transform_to_base_entities(llm_output)
        for entity in entities:
            if entity.text not in original_text:
                errors.append(
                    f"Validation Error: The extracted span '{entity.text}' was not found in the original text."
                )
        if errors:
            return {"validation_errors": errors, "validated_entities": None}
        return {"validation_errors": None, "validated_entities": entities}

    async def _refine_node(self, state: AgenticGraphState) -> dict:
        text_input = state["original_text"]
        previous_output = state["llm_output"]
        errors = state["validation_errors"]
        error_str = "\n- ".join(errors or [])
        previous_output_str = (
            json.dumps(previous_output.model_dump(), indent=2)
            if previous_output
            else "{}"
        )
        system_template = (
            "You are an extraction AI. You previously tried to extract entities but made mistakes. "
            "Review your previous output and the specific validation errors below, then try again. "
            "It is critical that you only extract text that is a verbatim substring of the source text.\n\n"
            "## Previous (Erroneous) Output:\n"
            "```json\n"
            "{previous_output}"
            "\n```\n\n"
            "## Validation Errors:\n"
            "- {errors}\n\n"
            "Please correct these errors and provide a new, valid JSON object."
        )
        human_template = "Source Text:\n```\n{text_input}\n```"
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_template), ("human", human_template)]
        )
        chain = prompt.partial(
            previous_output=previous_output_str, errors=error_str
        ) | self.model.with_structured_output(state["extraction_schema"].model_json_schema())
        result = await chain.ainvoke({"text_input": text_input})
        new_llm_output = state["extraction_schema"].model_validate(result)
        return {"llm_output": new_llm_output, "retry_count": state["retry_count"] + 1}

    def _decide_to_refine_or_end(self, state: AgenticGraphState) -> str:
        if state.get("validation_errors"):
            if state["retry_count"] < self.max_retries:
                return "refine"
        return END

    def _transform_to_base_entities(self, llm_output: BaseModel) -> List[BaseEntity]:
        base_entities: List[BaseEntity] = []
        if not llm_output:
            return base_entities
        for entity_type, extracted_spans in llm_output.model_dump().items():
            if not isinstance(extracted_spans, list):
                extracted_spans = [extracted_spans]
            for span_text in extracted_spans:
                if isinstance(span_text, str) and span_text:
                    base_entities.append(
                        BaseEntity(type=entity_type.capitalize(), text=span_text)
                    )
        return base_entities
