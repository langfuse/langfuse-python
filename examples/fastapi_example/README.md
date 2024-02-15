# fastapi_example

This is an example FastAPI application showcasing integration with Langfuse for event tracing and response generation.

1. **Shutdown Behavior**: The application defines shutdown logic using FastAPI's lifespan feature. On shutdown, it flushes all events to Langfuse, ensuring data integrity and completeness.

2. **Endpoints**:
   - `/`: Returns a simple message demonstrating the usage of Langfuse with FastAPI.
   - `"/campaign/"`: Accepts a `prompt` and employs Langfuse for event tracing. (Note: OpenAI is referenced for context but not used in this example).

3. **Integration**:
   - Langfuse: Utilized for event tracing with `trace`, `score`, `generation`, and `span` operations. (Note that OpenAI is not actually used here to generate an answer to the prompt. This example is just to show how to use FastAPI with the Langfuse SDK)

4. **Dependencies**:
   - FastAPI: Web framework for building APIs.
   - Langfuse: Library for event tracing and management.

5. **Usage**:
   - Preparation: Ensure langfuse is installed and configured in the `fastapi_example/main.py` file.
   - Starting the Server: Navigate to the root directory of the project `langfuse-python/examples/fastapi_examples`. Run the application using `poetry run start`.
   - Access endpoints at `http://localhost:8000`.

For more details on FastAPI and Langfuse refer to their respective documentation.
