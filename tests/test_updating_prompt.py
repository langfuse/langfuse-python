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
        prompt_name=prompt_name,
        prompt_version=1,
        new_labels=["john", "doe"],
    )

    # Fetch prompt after update (should be invalidated)
    fetched_prompt = langfuse.get_prompt(prompt_name)

    # Verify the fetched prompt matches the updated values
    assert fetched_prompt.name == prompt_name
    assert fetched_prompt.version == 1
    print(f"Fetched prompt labels: {fetched_prompt.labels}")
    print(f"Updated prompt labels: {updated_prompt.labels}")

    # production was set by the first call, latest is managed and set by Langfuse
    expected_labels = sorted(["latest", "doe", "production", "john"])
    assert sorted(fetched_prompt.labels) == expected_labels
    assert sorted(updated_prompt.labels) == expected_labels
