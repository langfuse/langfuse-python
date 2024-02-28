from datetime import datetime
from typing import Any, List, Optional, TypedDict, Literal

SpanLevel = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]


class TraceMetadata(TypedDict):
    name: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    version: Optional[str]
    release: Optional[str]
    metadata: Optional[Any]
    tags: Optional[List[str]]


class ObservationParams(TraceMetadata, TypedDict):
    input: Optional[Any]
    output: Optional[Any]
    level: Optional[SpanLevel]
    status_message: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
