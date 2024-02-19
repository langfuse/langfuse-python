# django_example

This Django application demonstrates integrating Langfuse for event tracing and response generation within a Django framework.

1. **Shutdown Behavior**: Implements shutdown logic using Django's framework. Shutdown, located in `myapp/__init__.py`, flushes all events to Langfuse to ensure data integrity.

2. **Endpoints**:
- `"/"`: Returns a JSON message to demonstrate Langfuse integration.
- `"/campaign/"`: Accepts a `prompt` and employs Langfuse for event tracing. (Note: OpenAI is referenced for context but not used in this example).

3. **Integration**:
- Langfuse: Utilized for event tracing with `trace`, `score`, `generation`, and `span` operations. (Note that OpenAI is not actually used here to generate an answer to the prompt. This example is just to show how to use FastAPI with the Langfuse SDK)

4. **Dependencies**:
- Django: The primary framework for building the application.
- Langfuse: Library for event tracing and management.

5. **Usage**:<br>
- Preparation: Ensure `langfuse` is installed and configured in the `myapp/langfuse_integration.py` file.<br>
- Starting the Server: Navigate to the root directory of the project `langfuse-python/examples/django_examples`. Run `poetry run python manage.py runserver 0.0.0.0:8000` to start the server.
- Accessing Endpoints: The application's endpoints can be accessed at `http://localhost:8000`.

Refer to Django and Langfuse documentation for more detailed information.
