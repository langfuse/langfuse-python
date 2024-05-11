# vllm_openai_example

This is an example of OpenAI compatible server based on vLLM and FastAPI showcasing integration with Langfuse for for event tracing and response generation.

1. **Shutdown Behavior**: The application defines shutdown logic using FastAPI's lifespan feature. On shutdown, it flushes all events to Langfuse, ensuring data integrity and completeness.

2. **Endpoints**:
   - `/health`: Returns healthcheck status
   - `/v1/models`: Shows available models
   - `/version`: Shows current version of vLLM
   - `/v1/chat/completions`: Generates chat completion and tracks sync and async responses to Langfuse
   - `/v1/completions`: Generates completion and tracks sync and async responses to Langfuse
   - More info can be found [here](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai)

3. **Integration**:
   - Langfuse: Utilized for event tracing with `@observe` decorator. It uses `transform_to_string` parameter to correctly process async results.

4. **Dependencies**:
   - Langfuse: Library for event tracing and management.
   - vLLM: vLLM is a fast and easy-to-use library for LLM inference and serving

5. **Usage**:
   - Preparation: Ensure that you meet requirements specified in [vllm repo](https://github.com/vllm-project/vllm) to setup running environment
   - Starting the Server: Navigate to the root directory of the project `langfuse-python/examples/vllm_openai_example`. Run the application using `python3 api_server.py --model <MODEL_PATH> --langfuse-public-key <PBK> --langfuse-secret-key <SCK> --langfuse-host <URL>`.
   - Access endpoints at `http://localhost:8000`.

Consider checking `api_server.py`, `cli_args.py`, `serving_chat.py`, `serving_competion.py` as they are modified version of vLLM code.
Also check `langfuse_utils.py` as it contains main logic for correctly processing `stream=True` requests from API.
For more details on vLLM and Langfuse refer to their respective documentation.
