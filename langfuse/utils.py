import uuid
import pydantic

from langfuse.input_validation import CreateGenerationValidation, UpdateGenerationValidation, UpdateSpanValidation


def convert_observation_to_event(body: pydantic.BaseModel, type: str, update: bool = False):
    dict_body = body.dict(exclude_none=True)
    dict_body["type"] = type

    if isinstance(body, CreateGenerationValidation) or isinstance(body, UpdateGenerationValidation):
        dict_body["output"] = body.completion
        dict_body.pop("completion", None)
        dict_body["input"] = body.prompt
        dict_body.pop("prompt", None)

    if isinstance(body, UpdateGenerationValidation):
        dict_body["id"] = body.generation_id
        dict_body.pop("generation_id", None)

    if isinstance(body, UpdateSpanValidation):
        dict_body["id"] = body.span_id
        dict_body.pop("span_id", None)

    return {
        "id": str(uuid.uuid4()),
        "type": "observation-update" if update else "observation-create",
        "body": dict_body,
    }
