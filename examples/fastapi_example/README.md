# fastapi_example

This is an example FastAPI application showcasing integration with Langfuse for event tracing and response generation.

1. **Startup and Shutdown Behavior**: The application defines startup and shutdown logic using FastAPI's lifespan feature. On startup, it initializes Langfuse for event tracing. On shutdown, it flushes all events to Langfuse, ensuring data integrity and completeness.

2. **Endpoints**:
   - `/`: Returns a simple message demonstrating the usage of Langfuse with FastAPI.
   - `/campaign/`: Accepts a prompt query parameter and utilizes OpenAI to generate a response. It also traces events using Langfuse.

3. **Integration**:
   - Langfuse: Utilized for event tracing with `trace`, `score`, `generation`, and `span` operations. (Note that OpenAI is not actually used here to generate an answer to the prompt. This example is just to show how to use FastAPI with the Langfuse SDK)

4. **Dependencies**:
   - FastAPI: Web framework for building APIs.
   - Langfuse: Library for event tracing and management.
   - OpenAI: AI platform for natural language processing.

5. **Usage**:
   - Start the application using `poetry run start` at the root level.
   - Access endpoints at `http://localhost:8000`.

For more details on FastAPI and Langfuse refer to their respective documentation.
