from langfuse import Langfuse

# Initialize Langfuse
langfuse = Langfuse(public_key="pk-lf-1234567890", secret_key="sk-lf-1234567890")


def get_response_openai(prompt):
    """
    This simulates the response to a prompt using the OpenAI API.

    Args:
        prompt (str): The prompt for generating the response.

    Returns:
        dict: A dictionary containing the response status and message (always "This is a test message").
    """
    try:
        trace = langfuse.trace(
            name="this-is-a-trace",
            user_id="test",
            metadata="test",
        )

        trace = trace.score(
            name="user-feedback",
            value=1,
            comment="Some user feedback",
        )

        generation = trace.generation(name="this-is-a-generation", metadata="test")

        sub_generation = generation.generation(
            name="this-is-a-sub-generation", metadata="test"
        )

        sub_sub_span = sub_generation.span(
            name="this-is-a-sub-sub-span", metadata="test"
        )

        sub_sub_span = sub_sub_span.score(
            name="user-feedback-o",
            value=1,
            comment="Some more user feedback",
        )

        response = {"status": "success", "message": "This is a test message"}
    except Exception as e:
        print("Error in creating campaigns from openAI:", str(e))
        return 503
    return response


def langfuse_flush():
    """Called by 'myapp/__init__.py' to flush any pending changes during shutdown."""
    langfuse.flush()
