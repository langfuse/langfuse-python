import uuid
import pydantic

from langfuse.api.resources.commons.types.create_generation_request import CreateGenerationRequest
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest


def convert_observation_to_event(body: pydantic.BaseModel, type: str, update: bool = False):
    dict_body = body.dict()
    dict_body["type"] = type

    if isinstance(body, CreateGenerationRequest) or isinstance(body, UpdateGenerationRequest):
        dict_body["output"] = body.completion
        dict_body.pop("completion", None)
        dict_body["input"] = body.prompt
        dict_body.pop("prompt", None)

    if isinstance(body, UpdateGenerationRequest):
        dict_body["id"] = body.generation_id
        dict_body.pop("generation_id", None)

    if isinstance(body, UpdateSpanRequest):
        dict_body["id"] = body.span_id
        dict_body.pop("span_id", None)

    return {
        "id": str(uuid.uuid4()),
        "type": "observation-update" if update else "observation-create",
        "body": dict_body,
    }
