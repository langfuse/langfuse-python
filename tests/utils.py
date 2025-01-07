import base64
import logging
import os
import typing
from uuid import uuid4

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

import backoff
import httpx
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager

from langfuse.api.client import FernLangfuse

logger = logging.getLogger(__name__)


def create_uuid():
    return str(uuid4())


class HTTPClientWithRetries(httpx.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=3,
        max_tries=4,
        giveup=lambda e: isinstance(e, httpx.HTTPStatusError)
        and e.response.status_code >= 500,
        on_backoff=lambda details: logger.warning(
            f"Request failed. Retrying in {details['wait']:.2f} seconds... "
            f"Attempt {details['tries']}/4"
        ),
    )
    def request(self, *args, **kwargs) -> httpx.Response:
        response = super().request(*args, **kwargs)
        response.raise_for_status()

        return response


def get_api():
    http_client_with_retries = HTTPClientWithRetries(timeout=10)

    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_HOST"),
        httpx_client=http_client_with_retries,
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
    if callback:
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


def encode_file_to_base64(image_path) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
