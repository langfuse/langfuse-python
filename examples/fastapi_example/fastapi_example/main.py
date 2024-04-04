from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, BackgroundTasks
from langfuse import Langfuse
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Operation on startup

    yield  # wait until shutdown

    # Flush all events to be sent to Langfuse on shutdown. This operation is blocking.
    langfuse.flush()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def main_route():
    return {
        "message": "Hey, this is an example showing how to use Langfuse with FastAPI."
    }


# Initialize Langfuse
langfuse = Langfuse(public_key="pk-lf-1234567890", secret_key="sk-lf-1234567890")


async def get_response_openai(prompt, background_tasks: BackgroundTasks):
    """This simulates the response to a prompt using the OpenAI API.

    Args:
        prompt (str): The prompt for generating the response.
        background_tasks (BackgroundTasks): An object for handling background tasks.

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


@app.get(
    "/campaign/",
    tags=["APIs"],
)
async def campaign(
    background_tasks: BackgroundTasks, prompt: str = Query(..., max_length=20)
):
    return await get_response_openai(prompt, background_tasks)


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("fastapi_example.main:app", host="0.0.0.0", port=8000, reload=True)
