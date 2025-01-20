from langfuse.client import Langfuse
from tests.utils import create_uuid


def test_update_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    # Create initial prompt
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
    )

    # Update prompt labels
    updated_prompt = langfuse.update_prompt(
        prompt_name=prompt_name, prompt_version=1, new_labels=["john", "doe"]
    )

    # Fetch prompt after update (should be invalidated)
    fetched_prompt = langfuse.get_prompt(prompt_name)

    # Verify the fetched prompt matches the updated values
    assert fetched_prompt.name == prompt_name
    assert fetched_prompt.version == 1
    assert fetched_prompt.labels == ["john", "doe"]
    assert updated_prompt.labels == ["john", "doe"]
