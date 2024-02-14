# django_example

This Django application demonstrates the integration of Langfuse for event tracing and response generation within a Django framework.

### Django Setup

1. **Project Creation**: A new Django project named `myproject` was created.
2. **App Creation**: Within this project, a Django app named `myapp` was established.
3. **Langfuse Installation**: The `langfuse` library was installed in the environment to facilitate event tracing.

### Django Views (`myapp/views.py`)

- **Main Route**: Returns a JSON message showcasing Langfuse usage in Django.
- **Campaign Route**: Handles requests with a `prompt` parameter and employs Langfuse for event tracing. Note: Length of the `prompt` is restricted to avoid errors.

### Langfuse Integration (`myapp/langfuse_integration.py`)

- **Configuration**: Langfuse is set up with public and secret keys.
- **OpenAI Response Generation**: Includes tracing, scoring, and handling generation events in Langfuse. Exception handling is implemented for robustness.
- **Flush Function**: Ensures all events are sent to Langfuse during shutdown.

### URL Routing (`myapp/urls.py`)

- Defines URL patterns for the main route and the campaign route, linking them to their respective views in `views.py`.

### Django Settings

- The `myapp` is included in the `INSTALLED_APPS` list in the project's `settings.py` file.

### Running the Server

- The server can be started using `poetry run python manage.py runserver 0.0.0.0:8000`.

### Shutdown Handling (`myapp/__init__.py`)

- A shutdown handler is implemented to flush data to Langfuse during application termination. This is crucial for ensuring data integrity and completeness.

For more details on Django and Langfuse refer to their respective documentation.