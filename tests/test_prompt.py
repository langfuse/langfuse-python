from langfuse.client import Langfuse
from tests.utils import get_api


def test_create_prompt():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test", prompt="test prompt", is_active=True
    )

    second_prompt_client = langfuse.get_prompt("test")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt


def test_compiling_prompt():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test",
        prompt="Hello, {{target}}! I hope you are {{state}}.",
        is_active=True,
    )

    second_prompt_client = langfuse.get_prompt("test")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt

    assert (
        second_prompt_client.compile({"target": "world", "state": "great"})
        == "Hello, world! I hope you are great."
    )


def test_prompt_end_to_end():
    langfuse = Langfuse(debug=False)

    langfuse.create_prompt(
        name="test",
        prompt="Hello, {{target}}! I hope you are {{state}}.",
        is_active=True,
    )

    prompt = langfuse.get_prompt("test")

    prompt_str = prompt.compile({"target": "world", "state": "great"})
    assert prompt_str == "Hello, world! I hope you are great."

    langfuse.generation(input=prompt_str, prompt=prompt)

    langfuse.flush()

    api = get_api()

    trace_id = langfuse.get_trace_id()

    trace = api.trace.get(trace_id)

    assert len(trace.observations) == 1

    generation = trace.observations[0]
    assert generation.prompt_id is not None

    observation = api.observations.get(generation.id)

    assert observation.prompt_id is not None
