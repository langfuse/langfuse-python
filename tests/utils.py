import os
import typing
from uuid import uuid4

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

from langfuse.api.client import FernLangfuse

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)

from llama_index.core.callbacks import CallbackManager


def create_uuid():
    return str(uuid4())


def get_api():
    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_HOST"),
    )


class LlmUsageWithCost(pydantic.BaseModel):
    prompt_tokens: typing.Optional[int] = pydantic.Field(
        alias="promptTokens", default=None
    )
    completion_tokens: typing.Optional[int] = pydantic.Field(
        alias="completionTokens", default=None
    )
    total_tokens: typing.Optional[int] = pydantic.Field(
        alias="totalTokens", default=None
    )
    input_cost: typing.Optional[float] = pydantic.Field(alias="inputCost", default=None)
    output_cost: typing.Optional[float] = pydantic.Field(
        alias="outputCost", default=None
    )
    total_cost: typing.Optional[float] = pydantic.Field(alias="totalCost", default=None)


class CompletionUsage(pydantic.BaseModel):
    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""


class LlmUsage(pydantic.BaseModel):
    prompt_tokens: typing.Optional[int] = pydantic.Field(
        alias="promptTokens", default=None
    )
    completion_tokens: typing.Optional[int] = pydantic.Field(
        alias="completionTokens", default=None
    )
    total_tokens: typing.Optional[int] = pydantic.Field(
        alias="totalTokens", default=None
    )

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().dict(**kwargs_with_defaults)


def get_llama_index_index(callback, force_rebuild: bool = False):
    Settings.callback_manager = CallbackManager([callback])
    PERSIST_DIR = "tests/mocks/llama-index-storage"

    if not os.path.exists(PERSIST_DIR) or force_rebuild:
        print("Building RAG index...")
        documents = SimpleDirectoryReader(
            "static", ["static/state_of_the_union_short.txt"]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("Using pre-built index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index
