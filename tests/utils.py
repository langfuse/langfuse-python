import base64
import os
import typing
from time import sleep
from uuid import uuid4

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

from langfuse.api.client import FernLangfuse


def create_uuid():
    return str(uuid4())


def get_api():
    sleep(3)

    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
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


def encode_file_to_base64(image_path) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
