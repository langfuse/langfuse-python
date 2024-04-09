# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import typing

from .chat_prompt import ChatPrompt
from .text_prompt import TextPrompt


class Prompt_Chat(ChatPrompt):
    type: typing.Literal["chat"] = "chat"

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True
        populate_by_name = True


class Prompt_Text(TextPrompt):
    type: typing.Literal["text"] = "text"

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True
        populate_by_name = True


Prompt = typing.Union[Prompt_Chat, Prompt_Text]
