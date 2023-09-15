from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler

from langfuse.api.resources.commons.types.llm_usage import LlmUsage
from langfuse.api.resources.commons.types.observation_level import ObservationLevel
from langfuse.client import Langfuse, StatefulClient, StatefulTraceClient
from langfuse.model import CreateGeneration, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan
from langchain.schema.output import LLMResult
from langchain.schema.messages import BaseMessage
from langchain.schema.document import Document

from langchain.schema.agent import AgentAction, AgentFinish


class Run:
    def __init__(self, state: StatefulClient, parent_id: Optional[str]) -> None:
        self.state = state
        self.parent_id = parent_id


class CallbackHandler(BaseCallbackHandler):
    log = logging.getLogger("langfuse")
    nextSpanId: Optional[str] = None

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        debug: bool = False,
        statefulTraceClient: Optional[StatefulTraceClient] = None,
        release: Optional[str] = None,
    ) -> None:
        # If we're provided a stateful trace client directly
        if statefulTraceClient:
            self.trace = Run(statefulTraceClient, None)
            self.runs = {}

        # Otherwise, initialize stateless using the provided keys
        elif public_key and secret_key:
            self.langfuse = Langfuse(public_key, secret_key, host, debug=debug, release=release)
            self.trace = None
            self.runs = {}
            if debug:
                # Ensures that debug level messages are logged when debug mode is on.
                # Otherwise, defaults to WARNING level.
                # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided
                logging.basicConfig()
                self.log.setLevel(logging.DEBUG)
            else:
                self.log.setLevel(logging.WARNING)
        else:
            self.log.error("Either provide a stateful langfuse object or both public_key and secret_key.")
            raise ValueError("Either provide a stateful langfuse object or both public_key and secret_key.")

    def flush(self):
        if self.trace is None:
            self.log.debug("There was no trace yet, hence no flushing possible.")

        self.trace.state.task_manager.flush()

    def setNextSpan(self, id: str):
        self.nextSpanId = id

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        # Nothing needs to happen here for langfuse. Once the streaming is done,
        self.log.debug(f"on llm new token: run_id: {run_id} parent_run_id: {parent_run_id}")

    def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors."""
        try:
            self.log.debug(f"on retriever error: run_id: {run_id} parent_run_id: {parent_run_id}")

            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(
                UpdateSpan(level=ObservationLevel.ERROR, statusMessage=str(error), endTime=datetime.now())
            )
        except Exception as e:
            self.log.exception(e)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on chain start: run_id: {run_id} parent_run_id: {parent_run_id}")
            self.__generate_trace_and_parent(
                serialized=serialized,
                inputs=inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                kwargs=kwargs,
            )
        except Exception as e:
            self.log.exception(e)

    def get_trace_id(self) -> str:
        return self.trace.state.id

    def __generate_trace_and_parent(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])

            if self.trace is None and self.langfuse is not None:
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
                            id=self.nextSpanId,
                            name=class_name,
                            metadata=self.__join_tags_and_metadata(tags, metadata),
                            input=inputs,
                            startTime=datetime.now(),
                        )
                    ),
                    parent_run_id,
                )
                self.nextSpanId = None
            else:
                self.runs[run_id] = Run(
                    self.trace.state.span(
                        CreateSpan(
                            id=self.nextSpanId,
                            name=class_name,
                            metadata=self.__join_tags_and_metadata(tags, metadata),
                            input=inputs,
                            startTime=datetime.now(),
                        )
                    ),
                    parent_run_id,
                )
                self.nextSpanId = None

        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(f"on agent action: run_id: {run_id} parent_run_id: {parent_run_id}")

            if run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(endTime=datetime.now(), output=action))
        except Exception as e:
            self.log.exception(e)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on agent finish: run_id: {run_id} parent_run_id: {parent_run_id}")
            if run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(endTime=datetime.now(), output=finish))
        except Exception as e:
            self.log.exception(e)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on chain end: run_id: {run_id} parent_run_id: {parent_run_id}")

            if run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(output=outputs, endTime=datetime.now()))
        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(f"on chain error: run_id: {run_id} parent_run_id: {parent_run_id}")
            self.runs[run_id].state = self.runs[run_id].state.update(
                UpdateSpan(level=ObservationLevel.ERROR, statusMessage=str(error), endTime=datetime.now())
            )
        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(f"on chat model start: run_id: {run_id} parent_run_id: {parent_run_id}")
            self.__on_llm_action(serialized, run_id, messages, parent_run_id, tags=tags, metadata=metadata, **kwargs)
        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(f"on llm start: run_id: {run_id} parent_run_id: {parent_run_id}")
            self.__on_llm_action(serialized, run_id, prompts, parent_run_id, tags=tags, metadata=metadata, **kwargs)
        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(f"on tool start: run_id: {run_id} parent_run_id: {parent_run_id}")

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")
            meta = self.__join_tags_and_metadata(tags, metadata)

            meta.update({key: value for key, value in kwargs.items() if value is not None})

            self.runs[run_id] = Run(
                self.runs[parent_run_id].state.span(
                    CreateSpan(
                        id=self.nextSpanId,
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        input=input_str,
                        startTime=datetime.now(),
                        metadata=meta,
                    )
                ),
                parent_run_id,
            )
            self.nextSpanId = None
        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(f"on retriever start: run_id: {run_id} parent_run_id: {parent_run_id}")

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")

            self.runs[run_id] = Run(
                self.runs[parent_run_id].state.span(
                    CreateSpan(
                        id=self.nextSpanId,
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        input=query,
                        startTime=datetime.now(),
                        metadata=self.__join_tags_and_metadata(tags, metadata),
                    )
                ),
                parent_run_id,
            )
            self.nextSpanId = None
        except Exception as e:
            self.log.exception(e)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on retriever end: run_id: {run_id} parent_run_id: {parent_run_id}")

            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(
                UpdateSpan(output=documents, endTime=datetime.now())
            )
        except Exception as e:
            self.log.exception(e)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on tool end: run_id: {run_id} parent_run_id: {parent_run_id}")
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(UpdateSpan(output=output, endTime=datetime.now()))
        except Exception as e:
            self.log.exception(e)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on tool error: run_id: {run_id} parent_run_id: {parent_run_id}")
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self.runs[run_id].state = self.runs[run_id].state.update(
                UpdateSpan(statusMessage=error, level=ObservationLevel.ERROR, endTime=datetime.now())
            )
        except Exception as e:
            self.log.exception(e)

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
                self.__generate_trace_and_parent(
                    serialized,
                    inputs=prompts,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    tags=tags,
                    metadata=metadata,
                    kwargs=kwargs,
                )
            if kwargs["invocation_params"]["_type"] in ["anthropic-llm", "anthropic-chat"]:
                model_name = "anthropic"  # unfortunately no model info by anthropic provided.
            elif kwargs["invocation_params"]["_type"] == "huggingface_hub":
                model_name = kwargs["invocation_params"]["repo_id"]
            elif kwargs["invocation_params"]["_type"] == "azure-openai-chat":
                model_name = kwargs["invocation_params"]["model"]
            elif kwargs["invocation_params"]["_type"] == "llamacpp":
                model_name = kwargs["invocation_params"]["model_path"]
            else:
                model_name = kwargs["invocation_params"]["model_name"]
            self.runs[run_id] = Run(
                self.runs[parent_run_id].state.generation(
                    CreateGeneration(
                        name=serialized.get("name", serialized.get("id", ["<unknown>"])[-1]),
                        prompt=prompts,
                        startTime=datetime.now(),
                        metadata=self.__join_tags_and_metadata(tags, metadata),
                        model=model_name,
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
                        model=model_name,
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
            self.log.exception(e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on llm end: run_id: {run_id} parent_run_id: {parent_run_id} response: {response} kwargs: {kwargs}"
            )
            if run_id not in self.runs:
                raise Exception("run not found")
            else:
                last_response = response.generations[-1][-1]
                llm_usage = None if response.llm_output is None else LlmUsage(**response.llm_output["token_usage"])

                extracted_response = (
                    last_response.text
                    if last_response.generation_info is None
                    or (
                        "finish_reason" not in last_response.generation_info
                        or last_response.generation_info["finish_reason"] != "function_call"
                    )
                    else str(last_response.message.additional_kwargs)
                )

                self.runs[run_id].state = self.runs[run_id].state.update(
                    UpdateGeneration(completion=extracted_response, end_time=datetime.now(), usage=llm_usage)
                )
        except Exception as e:
            self.log.exception(e)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(f"on llm error: run_id: {run_id} parent_run_id: {parent_run_id}")
            self.runs[run_id].state = self.runs[run_id].state.update(
                UpdateGeneration(endTime=datetime.now(), statusMessage=str(error), level=ObservationLevel.ERROR)
            )
        except Exception as e:
            self.log.exception(e)

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
