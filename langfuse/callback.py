from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langfuse.api.model import CreateGeneration, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan
from langfuse.api.resources.span.types.observation_level_span import ObservationLevelSpan
from langchain.schema.output import LLMResult

from langfuse.client import Langfuse, StatefulSpanClient


class CallbackHandler(BaseCallbackHandler):
    def __init__(self, langfuse: Langfuse) -> None:
        self.langfuse = langfuse

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        try:
            class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
            print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m", serialized, inputs, run_id, parent_run_id, tags, kwargs)

            trace = self.langfuse.trace(
                CreateTrace(
                    name=f"{class_name}-{run_id}",
                    metadata=metadata,
                )
            )
            self.trace = trace

            span = trace.span(
                CreateSpan(
                    name=f"{class_name}-{run_id}",
                    metadata=metadata,
                    input=inputs,
                )
            )
            self.parentSpan = span
            self.client = span

        except Exception as e:
            print(e)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            print("\n\033[1m> Finished chain.\033[0m", outputs, run_id, parent_run_id, kwargs)
            if isinstance(self.parentSpan, StatefulSpanClient):
                self.parentSpan = self.parentSpan.update(UpdateSpan(output=outputs))
                self.langfuse.flush()
            else:
                raise Exception("unexpected error")
        except Exception as e:
            print(e)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            print("\n\033[1m> Errored chain.\033[0m", error, run_id, parent_run_id, tags, kwargs)

            self.client = self.parentSpan.update(UpdateSpan(level=ObservationLevelSpan.ERROR, statusMessage=str(error)))
            self.langfuse.flush()
        except Exception as e:
            print(e)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            print("\n\033[1m> LLM start.\033[0m", serialized, prompts, kwargs)
            self.client = self.client.generation(CreateGeneration(name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]), prompt=prompts))
        except Exception as e:
            print(e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            print("\n\033[1m> LLM end.\033[0m", response, run_id, parent_run_id, kwargs)
            last_response = response.generations[-1][-1].text
            self.client = self.client.update(UpdateGeneration(completion=last_response))
        except Exception as e:
            print(e)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print("\n\033[1m> LLM error.\033[0m", error, run_id, parent_run_id, kwargs)
