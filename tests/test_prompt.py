from langfuse.client import Langfuse


def test_create_prompt():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test", prompt="test prompt", is_active=True
    )

    second_prompt_client = langfuse.get_prompt("test")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
