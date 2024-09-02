# This file was auto-generated by Fern from our API Definition.

from .session import Session
import typing
from .trace import Trace
from ....core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class SessionWithTraces(Session):
    traces: typing.List[Trace]

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
            extra="allow", frozen=True
        )  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
