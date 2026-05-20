import openai

from langfuse._client.client import Langfuse
from tests.support.utils import create_uuid


def test_create_chat_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "test prompt 1 with {{animal}}"},
            {"role": "user", "content": "test prompt 2 with {{occupation}}"},
        ],
        labels=["production"],
        tags=["test"],
        type="chat",
        commit_message="initial commit",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, type="chat")

    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=prompt_client.compile(animal="dog", occupation="doctor"),
    )

    assert len(completion.choices) > 0

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.labels == ["production", "latest"]
    assert prompt_client.tags == second_prompt_client.tags
    assert prompt_client.commit_message == second_prompt_client.commit_message
    assert prompt_client.config == {}


def test_create_chat_prompt_with_placeholders():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "You are a {{role}} assistant"},
            {"type": "placeholder", "name": "history"},
            {"role": "user", "content": "Help me with {{task}}"},
        ],
        labels=["production"],
        tags=["test"],
        type="chat",
        commit_message="initial commit",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, type="chat")
    messages = second_prompt_client.compile(
        role="helpful",
        task="coding",
        history=[
            {"role": "user", "content": "Example: {{task}}"},
            {"role": "assistant", "content": "Example response"},
        ],
    )

    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    assert len(completion.choices) > 0
    assert len(messages) == 4
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["content"] == "Example: coding"
    assert messages[2]["content"] == "Example response"
    assert messages[3]["content"] == "Help me with coding"

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.labels == ["production", "latest"]
    assert prompt_client.tags == second_prompt_client.tags
    assert prompt_client.commit_message == second_prompt_client.commit_message
    assert prompt_client.config == {}
