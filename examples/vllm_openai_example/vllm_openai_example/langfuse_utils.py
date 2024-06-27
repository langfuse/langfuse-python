import pydantic
from typing import Iterable, Any, Dict
from functools import partial
from protocol import ChatCompletionStreamResponse, CompletionStreamResponse


def base_langfuse_stream_processor(
    generator: Iterable[str], _streaming_model: pydantic.BaseModel = None
) -> Dict[str, Any]:
    result_content = ""
    result_responses = []
    # Saving whole response takes more time than only saving `result_content`
    # Consider remove saving whole response if this time is significant for you
    for item in generator:
        result_responses.append(item)
        raw_object = item.rsplit("data: ", 1)[1]
        try:
            single_response = _streaming_model.model_validate_json(raw_object)
            content = single_response.choices[0].delta.content
        except pydantic.ValidationError:  # receive 'data: [DONE]\n\n'
            content = ""
        result_content += content if content is not None else ""

    return {"content": result_content, "data": result_responses}


langfuse_chat_transformer = partial(
    base_langfuse_stream_processor, _streaming_model=ChatCompletionStreamResponse
)
langfuse_completion_transformer = partial(
    base_langfuse_stream_processor, _streaming_model=CompletionStreamResponse
)
