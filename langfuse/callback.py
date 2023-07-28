from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langfuse.api.model import CreateGeneration, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan
from langfuse.api.resources.generations.types.llm_usage import LlmUsage
from langfuse.api.resources.span.types.observation_level_span import ObservationLevelSpan
from langchain.schema.output import LLMResult
from langchain.schema.messages import BaseMessage
from langchain.schema.document import Document
from langfuse.client import Langfuse, StatefulClient
from langchain.schema.agent import AgentAction, AgentFinish


class Run:
    def __init__(self, state: StatefulClient, parent_id: Optional[str]) -> None:
        self.state = state
        self.parent_id = parent_id


class CallbackHandler(BaseCallbackHandler):
    def __init__(self, public_key: str, secret_key: str, host: Optional[str]) -> None:
        self.langfuse = Langfuse(public_key, secret_key, host)
        self.trace = None
        self.runs = {}

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
            logging.debug(f"on chain start: {run_id}")
            self.__generate_trace_and_parent(serialized=serialized, inputs=inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, kwargs=kwargs)
        except Exception as e:
            logging.error(e)

    def __generate_trace_and_parent(
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

            if self.trace is None:
                trace = Run(
                    self.langfuse.trace(
                        CreateTrace(
                            name=class_name,
                            metadata=self.__join_tags_and_metadata(tags, metadata),
                        )
                    ),
                    None,
                )
                self.trace = trace

            if parent_run_id is not None and parent_run_id in self.runs:
                self.runs[run_id] = Run(
                    self.runs[parent_run_id].state.span(
                        CreateSpan(
                            name=class_name,
                            metadata=self.__join_tags_and_metadata(tags, metadata),
                            input=inputs,
                            startTime=datetime.now(),
                        )
                    ),
                    parent_run_id,
                )
            else:
                self.runs[run_id] = Run(
                    self.trace.state.span(
                        CreateSpan(
                            name=class_name,
                            metadata=self.__join_tags_and_metadata(tags, metadata),
                            input=inputs,
                            startTime=datetime.now(),
                        )
                    ),
                    parent_run_id,
                )

        except Exception as e:
            logging.error(e)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        try:
            logging.debug(f"on agent action: {run_id}")

            if run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(endTime=datetime.now(), output=action))
        except Exception as e:
            logging.error(e)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on agent finish: {run_id}")
            if run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(endTime=datetime.now(), output=finish))
        except Exception as e:
            logging.error(e)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on chain end: {run_id}")

            if run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(output=outputs, endTime=datetime.now()))
        except Exception as e:
            logging.error(e)

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
            logging.debug(f"on chain error: {run_id}")
            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(level=ObservationLevelSpan.ERROR, statusMessage=str(error), endTime=datetime.now()))
        except Exception as e:
            logging.error(e)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on chat model start: {run_id}")
            self.__on_llm_action(serialized, run_id, messages, parent_run_id, tags=tags, metadata=metadata, **kwargs)
        except Exception as e:
            logging.error(e)

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
            logging.debug(f"on llm start: {run_id}")
            self.__on_llm_action(serialized, run_id, prompts, parent_run_id, tags=tags, metadata=metadata, **kwargs)
        except Exception as e:
            logging.error(e)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on tool start: {run_id}")

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")
            meta = self.__join_tags_and_metadata(tags, metadata)

            meta.update({key: value for key, value in kwargs.items() if value is not None})

            self.runs[run_id] = Run(
                self.runs[parent_run_id].state.span(
                    CreateSpan(
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        input=input_str,
                        startTime=datetime.now(),
                        metadata=meta,
                    )
                ),
                parent_run_id,
            )
        except Exception as e:
            logging.error(e)

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on retriever start: {run_id}")

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")

            self.runs[run_id] = Run(
                self.runs[parent_run_id].state.span(
                    CreateSpan(
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        input=query,
                        startTime=datetime.now(),
                        metadata=self.__join_tags_and_metadata(tags, metadata),
                    )
                ),
                parent_run_id,
            )
        except Exception as e:
            logging.error(e)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on retriever end: {run_id}")

            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(output=documents, endTime=datetime.now()))
        except Exception as e:
            logging.error(e)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on tool end: {run_id}")
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(output=output, endTime=datetime.now()))
        except Exception as e:
            logging.error(e)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on tool error: {run_id}")
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(statusMessage=error, level=ObservationLevelSpan.ERROR, endTime=datetime.now()))
        except Exception as e:
            logging.error(e)

    def __on_llm_action(
        self,
        serialized: Dict[str, Any],
        run_id: UUID,
        prompts: List[str],
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            if self.trace is None:
                # simple LLM call that has no trace and parent
                self.__generate_trace_and_parent(serialized, inputs=prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, kwargs=kwargs)

            self.runs[run_id] = Run(
                self.runs[parent_run_id].state.generation(
                    CreateGeneration(
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        prompt=prompts,
                        startTime=datetime.now(),
                        metadata=self.__join_tags_and_metadata(tags, metadata),
                        model=kwargs["invocation_params"]["model_name"],
                        modelParameters={
                            key: value
                            for key, value in {
                                "temperature": kwargs["invocation_params"].get("temperature"),
                                "max_tokens": kwargs["invocation_params"].get("max_tokens"),
                                "top_p": kwargs["invocation_params"].get("top_p"),
                                "frequency_penalty": kwargs["invocation_params"].get("frequency_penalty"),
                                "presence_penalty": kwargs["invocation_params"].get("presence_penalty"),
                                "request_timeout": kwargs["invocation_params"].get("request_timeout"),
                            }.items()
                            if value is not None
                        },
                    )
                )
                if parent_run_id in self.runs
                else self.trace.state.generation(
                    CreateGeneration(
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        prompt=prompts,
                        startTime=datetime.now(),
                        metadata=self.__join_tags_and_metadata(tags, metadata),
                        model=kwargs["invocation_params"]["model_name"],
                        modelParameters={
                            key: value
                            for key, value in {
                                "temperature": kwargs["invocation_params"].get("temperature"),
                                "max_tokens": kwargs["invocation_params"].get("max_tokens"),
                                "top_p": kwargs["invocation_params"].get("top_p"),
                                "frequency_penalty": kwargs["invocation_params"].get("frequency_penalty"),
                                "presence_penalty": kwargs["invocation_params"].get("presence_penalty"),
                                "request_timeout": kwargs["invocation_params"].get("request_timeout"),
                            }.items()
                            if value is not None
                        },
                    )
                ),
                datetime.now(),
            )
        except Exception as e:
            logging.error(e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on llm end: {run_id}")
            if run_id not in self.runs:
                raise Exception("run not found")
            else:
                last_response = response.generations[-1][-1].text
                self.runs[run_id].state = self.runs[run_id].state.update(UpdateGeneration(completion=last_response, endTime=datetime.now(), usage=LlmUsage(**response.llm_output["token_usage"])))
        except Exception as e:
            logging.error(e)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.debug(f"on llm error: {run_id}")
            self.runs[run_id].state = self.runs[run_id].state.update(UpdateGeneration(endTime=datetime.now(), statusMessage=str(error), level=ObservationLevelSpan.ERROR))
        except Exception as e:
            logging.error(e)

    def __join_tags_and_metadata(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if tags is None and metadata is None:
            return None
        elif tags is not None and len(tags) > 0:
            final_dict = {"tags": tags}
            if metadata is not None:
                final_dict.update(metadata)  # Merge metadata into final_dict
            return final_dict
        else:
            return metadata
