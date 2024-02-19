import datetime as dt
import logging
import os
import typing
import uuid
import httpx
from enum import Enum
from typing import Literal, Optional

from langfuse.api.resources.ingestion.types.create_event_body import CreateEventBody
from langfuse.api.resources.ingestion.types.create_generation_body import (
    CreateGenerationBody,
)
from langfuse.api.resources.ingestion.types.create_span_body import CreateSpanBody
from langfuse.api.resources.ingestion.types.score_body import ScoreBody
from langfuse.api.resources.ingestion.types.trace_body import TraceBody
from langfuse.api.resources.ingestion.types.update_generation_body import (
    UpdateGenerationBody,
)
from langfuse.api.resources.ingestion.types.update_span_body import UpdateSpanBody
from langfuse.api.resources.prompts.types.create_prompt_request import (
    CreatePromptRequest,
)
from langfuse.model import (
    CreateDatasetItemRequest,
    CreateDatasetRequest,
    CreateDatasetRunItemRequest,
    DatasetItem,
    DatasetRun,
    DatasetStatus,
    ModelUsage,
    PromptClient,
)
from langfuse.prompt_cache import PromptCache

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

from langfuse.api.client import FernLangfuse
from langfuse.environment import get_common_release_envs
from langfuse.logging import clean_logger
from langfuse.model import Dataset, MapValue, Observation, TraceWithFullDetails
from langfuse.request import LangfuseClient
from langfuse.task_manager import TaskManager
from langfuse.utils import _convert_usage_input, _create_prompt_context, _get_timestamp

from .version import __version__ as version

class Langfuse(object):
    """Langfuse Python client that needs to be initialized to use Langfuse.

    Args:
        public_key (str, optional): Public API key of Langfuse project. Can be set via `LANGFUSE_PUBLIC_KEY` environment variable.
        secret_key (str, optional): Secret API key of Langfuse project. Can be set via `LANGFUSE_SECRET_KEY` environment variable.
        host (str, optional): Host of Langfuse API. Can be set via `LANGFUSE_HOST` environment variable. Defaults to `https://cloud.langfuse.com`.
        release (str, optional): Release number/hash of the application to provide analytics grouped by release. Can be set via `LANGFUSE_RELEASE` environment variable.
        debug (bool, optional): Enables debug mode for more verbose logging. Can be set via `LANGFUSE_DEBUG` environment variable. Defaults to False.
        threads (int, optional): Number of consumer threads to execute network requests. Helps scaling the SDK for high load. Only increase this if you run into scaling issues. Defaults to 1.
        flush_at (int, optional): Max batch size that's sent to the API. Defaults to 50.
        flush_interval (int, optional): Max delay until a new batch is sent to the API. Defaults to 0.5.
        max_retries (int, optional): Max number of retries in case of API/network errors. Defaults to 3.
        timeout (int, optional): Timeout of API requests in seconds. Defaults to 15.
        sdk_integration (str, optional): Used by intgerations that wrap the Langfuse SDK to add context for debugging and support. Not to be used directly. Defaults to "default".
        httpx_client (httpx.Client, optional): Pass your own httpx client for more customizability of requests. Defaults to None.

    Attributes:
        log (logging.Logger): Logger for the Langfuse client.
        host (str): Host address for the Langfuse API, defining the destination for API requests.
        base_url (str): Base URL of the Langfuse API, serving as the root address for API endpoint construction.
        httpx_client (httpx.Client): HTTPX client utilized for executing requests to the Langfuse API.
        client (FernLangfuse): Core interface for Langfuse API interaction.
        task_manager (TaskManager): Task Manager dedicated to handling asynchronous tasks.
        trace_id (str): Identifier of currently used trace.
        release (str): Identifies the release number or hash of the application.
        prompt_cache (PromptCache): A cache for efficiently storing and retrieving PromptClient instances.

    Raises:
        ValueError: If public_key or secret_key is not set and not found in environment variables.

    Example: Initiating the Langfuse client should always be first step to use Langfuse.
        ```python
        import os
        from langfuse import Langfuse

        # Set the public and secret keys as environment variables
        os.environ['LANGFUSE_PUBLIC_KEY'] = public_key
        os.environ['LANGFUSE_SECRET_KEY'] = secret_key

        # Initialize the Langfuse client using the credentials
        langfuse = Langfuse()
        ```
    """
    log = logging.getLogger("langfuse")
    host: str #Host of Langfuse API

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: bool = False,
        threads: int = 1,
        flush_at: int = 50,
        flush_interval: int = 0.5,
        max_retries=3,
        timeout=15,
        sdk_integration: str = "default",
        httpx_client: Optional[httpx.Client] = None,
    ):
        set_debug = debug if debug else (os.getenv("LANGFUSE_DEBUG", "False") == "True")

        if set_debug is True:
            # Ensures that debug level messages are logged when debug mode is on.
            # Otherwise, defaults to WARNING level.
            # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided
            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)

            clean_logger()
        else:
            self.log.setLevel(logging.WARNING)
            clean_logger()

        public_key = public_key if public_key else os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key if secret_key else os.environ.get("LANGFUSE_SECRET_KEY")
        self.base_url = (
            host
            if host
            else os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

        if not public_key:
            self.log.warning("public_key is not set.")
            raise ValueError(
                "public_key is required, set as parameter or environment variable 'LANGFUSE_PUBLIC_KEY'"
            )

        if not secret_key:
            self.log.warning("secret_key is not set.")
            raise ValueError(
                "secret_key is required, set as parameter or environment variable 'LANGFUSE_SECRET_KEY'"
            )

        self.httpx_client = (
            httpx.Client(timeout=timeout) if httpx_client is None else httpx_client
        )

        self.client = FernLangfuse(
            base_url=self.base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=version,
            x_langfuse_public_key=public_key,
            httpx_client=self.httpx_client,
        )

        langfuse_client = LangfuseClient(
            public_key=public_key,
            secret_key=secret_key,
            base_url=self.base_url,
            version=version,
            timeout=timeout,
            session=self.httpx_client,
        )

        args = {
            "threads": threads,
            "flush_at": flush_at,
            "flush_interval": flush_interval,
            "max_retries": max_retries,
            "client": langfuse_client,
            "public_key": public_key,
            "sdk_name": "python",
            "sdk_version": version,
            "sdk_integration": sdk_integration,
        }

        if threads is not None:
            args["threads"] = threads

        self.task_manager = TaskManager(**args)

        self.trace_id = None

        self.release = self.__get_release_value(release)

        self.prompt_cache = PromptCache()

    def __get_release_value(self, release: Optional[str] = None) -> Optional[str]:
        """Retrieve the release value of Langfuse.

        If 'release' is not provided, searches for a release identifier in a predefined set of environment variables.

        Args:
            release (Optional[str]): The explicit release identifier, if available.

        Returns:
            Optional[str]: The resolved release identifier, or None if not found.
        """
        if release:
            return release
        elif "LANGFUSE_RELEASE" in os.environ:
            return os.environ["LANGFUSE_RELEASE"]
        else:
            return get_common_release_envs()

    def get_trace_id(self) -> str:
        """Get the current trace id."""
        return self.trace_id

    def get_trace_url(self) -> str:
        """Get the URL to see the current trace in the Langfuse UI."""
        return f"{self.base_url}/trace/{self.trace_id}"

    def get_dataset(self, name: str):
        """Get the dataset client of the dataset with the given name."""
        try:
            self.log.debug(f"Getting datasets {name}")
            dataset = self.client.datasets.get(dataset_name=name)

            items = [DatasetItemClient(i, langfuse=self) for i in dataset.items]

            return DatasetClient(dataset, items=items)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_dataset_item(self, id: str):
        """Get the dataset item with the given id."""
        try:
            self.log.debug(f"Getting dataset item {id}")
            dataset_item = self.client.dataset_items.get(id=id)
            return DatasetItemClient(dataset_item, langfuse=self)
        except Exception as e:
            self.log.exception(e)
            raise e

    def auth_check(self) -> bool:
        """Check if the provided credentials (public and secret key) are valid.

        Raises:
            Exception: If no projects were found for the provided credentials.
        """
        try:
            projects = self.client.projects.get()
            self.log.debug(
                f"Auth check successful, found {len(projects.data)} projects"
            )
            if len(projects.data) == 0:
                raise Exception(
                    "Auth check failed, no project found for the keys provided."
                )
            return True

        except Exception as e:
            self.log.exception(e)
            raise e

    def get_dataset_run(
        self,
        dataset_name: str,
        dataset_run_name: str,
    ) -> DatasetRun:
        """Get the dataset run of a given dataset with a given name."""
        try:
            self.log.debug(
                f"Getting dataset runs for dataset {dataset_name} and run {dataset_run_name}"
            )
            return self.client.datasets.get_runs(
                dataset_name=dataset_name, run_name=dataset_run_name
            )
        except Exception as e:
            self.log.exception(e)
            raise e

    def create_dataset(self, name: str) -> Dataset:
        """Create a dataset with the given name on Langfuse."""
        try:
            body = CreateDatasetRequest(name=name)
            self.log.debug(f"Creating datasets {body}")
            return self.client.datasets.create(request=body)
        except Exception as e:
            self.log.exception(e)
            raise e

    def create_dataset_item(
        self,
        dataset_name: str,
        input: any,
        expected_output: Optional[any] = None,
        id: Optional[str] = None,
    ) -> DatasetItem:
        """Create a dataset item.
        
        Upserts if an item with id already exists.

        Args:
            dataset_name (str): Name of the dataset in which the dataset item should be created.
            input (any): Input data. Can contain any python object or value.
            expected_output (Optional[any]): Expected output data. Defaults to None.
            id (Optional[str]): Id of the dataset item. Defaults to None.

        Returns:
            DatasetItem: The created dataset item.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Uploading items to the Langfuse dataset named "capital_cities"
            langfuse.create_dataset_item(
                dataset_name="capital_cities",
                input={"input": {"country": "Italy"}},
                expected_output={"expected_output": "Rome"}
            ```
        """
        try:
            body = CreateDatasetItemRequest(
                datasetName=dataset_name,
                input=input,
                expectedOutput=expected_output,
                id=id,
            )
            self.log.debug(f"Creating dataset item {body}")
            return self.client.dataset_items.create(request=body)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_trace(
        self,
        id: str,
    ) -> TraceWithFullDetails:
        """Get a trace in the current project with the given identifier."""
        try:
            self.log.debug(f"Getting trace {id}")
            return self.client.trace.get(id)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_observations(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        type: typing.Optional[str] = None,
    ):
        """Get a list of observations in the current project matching the given parameters.

        Args:
            page (Optional[int]): Page number of the observations to return. Defaults to None.
            limit (Optional[int]): Maximum number of observations to return. Defaults to None.
            name (Optional[str]): Name of the observations to return. Defaults to None.
            user_id (Optional[str]): User identifier. Defaults to None.
            trace_id (Optional[str]): Trace identifier. Defaults to None.
            parent_observation_id (Optional[str]): Parent observation identifier. Defaults to None.
            type (Optional[str]): Type of the observation. Defaults to None.

        Returns:
            List of ObservationsViews: List of observations in the project matching the given parameters.
        """
        try:
            self.log.debug(
                f"Getting observations... {page}, {limit}, {name}, {user_id}, {trace_id}, {parent_observation_id}, {type}"
            )
            return self.client.observations.get_many(
                page=page,
                limit=limit,
                name=name,
                user_id=user_id,
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                type=type,
            )
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_generations(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
    ):
        """Get a list of generations in the current project matching the given parameters.

        Args:
            page (Optional[int]): Page number of the generations to return. Defaults to None.
            limit (Optional[int]): Maximum number of generations to return. Defaults to None.
            name (Optional[str]): Name of the generations to return. Defaults to None.
            user_id (Optional[str]): User identifier of the generations to return. Defaults to None.
            trace_id (Optional[str]): Trace identifier of the generations to return. Defaults to None.
            parent_observation_id (Optional[str]): Parent observation identifier of the generations to return. Defaults to None.

        Returns:
            List of ObservationsViews: List of geneations in the project matching the given parameters.
        """
        return self.get_observations(
            page=page,
            limit=limit,
            name=name,
            user_id=user_id,
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            type="GENERATION",
        )

    def get_observation(
        self,
        id: str,
    ) -> Observation:
        """Get an observation in the current project with the given identifier."""
        try:
            self.log.debug(f"Getting observation {id}")
            return self.client.observations.get(id)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_prompt(
        self,
        name: str,
        version: Optional[int] = None,
        *,
        cache_ttl_seconds: Optional[int] = None,
    ) -> PromptClient:
        """Get a prompt by its name and optionally its version, with support for additional options.

        This method attempts to fetch the requested prompt from the local cache. If the prompt is not found
        in the cache or if the cached prompt has expired, it will try to fetch the prompt from the server again
        and update the cache. If fetching the new prompt fails, and there is an expired prompt in the cache, it will
        return the expired prompt as a fallback.

        Args:
            name (str): The name of the prompt to retrieve.
            version (Optional[int]): The version of the prompt. If not specified, the latest version is assumed.
            cache_ttl_seconds: Optional[int]: Time-to-live in seconds for caching the prompt. Must be specified as a
            keyword argument. If specified, a default of 60 seconds is used.

        Returns:
            PromptClient: The prompt object retrieved from the cache or directly fetched if not cached or expired.

        Raises:
            Exception: Propagates any exceptions raised during the fetching of a new prompt, unless there is an
            expired prompt in the cache, in which case it logs a warning and returns the expired prompt.
        """
        self.log.debug(f"Getting prompt {name}, version {version or 'latest'}")

        if not name:
            raise ValueError("Prompt name cannot be empty.")

        cache_key = PromptCache.generate_cache_key(name, version)
        cached_prompt = self.prompt_cache.get(cache_key)

        if cached_prompt is None:
            return self._fetch_prompt_and_update_cache(name, version, cache_ttl_seconds)

        if cached_prompt.is_expired():
            try:
                return self._fetch_prompt_and_update_cache(
                    name, version, cache_ttl_seconds
                )

            except Exception as e:
                self.log.warn(
                    f"Returning expired prompt cache for '${name}-${version or 'latest'}' due to fetch error: {e}"
                )

                return cached_prompt.value

        return cached_prompt.value

    def _fetch_prompt_and_update_cache(
        self,
        name: str,
        version: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> PromptClient:
        """Fetch a prompt from the server based on its name and version, and update the prompt cache.

        Args:
            name (str): The name of the prompt to retrieve.
            version (Optional[int]): The version of the prompt to retrieve.
            If not specified, the latest version is assumed. Defaults to None.
            ttl_seconds (Optional[int]): The time-to-live in seconds for the prompt in the cache.
            If not specified, the prompt will not have an expiration in the cache. Defaults to None.

        Raises:
            Exception: Propagates any exceptions raised during the fetching of the prompt from the server.

        Returns:
            PromptClient: The prompt object fetched from the server.
        """
        try:
            self.log.debug(
                f"Fetching prompt {name}-{version or 'latest'}' from server..."
            )

            promptResponse = self.client.prompts.get(name=name, version=version)
            cache_key = PromptCache.generate_cache_key(name, version)
            prompt = PromptClient(promptResponse)

            self.prompt_cache.set(cache_key, prompt, ttl_seconds)

            return prompt

        except Exception as e:
            self.log.exception(
                f"Error while fetching prompt '{name}-{version or 'latest'}': {e}"
            )

            raise e

    def create_prompt(self, *, name: str, prompt: str, is_active: bool) -> PromptClient:
        """Create a new prompt with the specified name, prompt content, and active status.

        Args:
            name (str): The name of the prompt to be created.
            prompt (str): The content of the prompt to be created.
            is_active (bool): A flag indicating whether the prompt is active or not.
            Active prompts are can be retrieved via the SDK and monitored during usage.

        Returns:
            PromptClient: Prompt client representing the prompt.
        """
        try:
            self.log.debug(f"Creating prompt {name}, version {version}")

            request = CreatePromptRequest(
                name=name,
                prompt=prompt,
                is_active=is_active,
            )
            prompt = self.client.prompts.create(request=request)
            return PromptClient(prompt=prompt)
        except Exception as e:
            self.log.exception(e)
            raise e

    def trace(
        self,
        *,
        id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        metadata: typing.Optional[typing.Any] = None,
        tags: typing.Optional[typing.List[str]] = None,
        **kwargs,
    ):
        """Create a trace with the provided details.

        This method is used to create a new trace, which includes various optional details like name, user ID,
        version, input, output, metadata, and tags. It generates a unique identifier for the trace if not provided,
        timestamps it, and queues it for processing by the task manager.

        Args:
            id (Optional[str]): A unique identifier for the trace. If not provided, a new UUID is generated.
            name (Optional[str]): The name of the trace.
            user_id (Optional[str]): The user ID of the trace.
            version (Optional[str]): The version of the trace.
            input (Optional[Any]): The input data of the trace.
            output (Optional[Any]): The output data of the trace.
            metadata (Optional[Any]): Additional metadata of the trace.
            tags (Optional[List[str]]): A list of tags for categorizing or labeling the trace.
            **kwargs: Additional keyword arguments that can be included in the trace .

        Returns:
            StatefulTraceClient: The created trace.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Creating a trace for a text summarization generation
            trace = langfuse.trace(
                name="example-application", 
                user_id="user-1234")
            )
            ```
        """
        try:
            new_id = str(uuid.uuid4()) if id is None else id
            self.trace_id = new_id

            new_dict = {
                "id": new_id,
                "name": name,
                "userId": user_id,
                "release": self.release,
                "version": version,
                "metadata": metadata,
                "input": input,
                "output": output,
                "tags": tags,
                "timestamp": _get_timestamp(),
            }
            if kwargs is not None:
                new_dict.update(kwargs)

            new_body = TraceBody(**new_dict)

            self.log.debug(f"Creating trace {new_body}")
            event = {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "body": new_body.dict(exclude_none=True),
            }

            self.task_manager.add_task(
                event,
            )

            return StatefulTraceClient(
                self.client, new_id, StateType.TRACE, new_id, self.task_manager
            )
        except Exception as e:
            self.log.exception(e)

    def score(
        self,
        *,
        name: str,
        value: float,
        trace_id: typing.Optional[str] = None,
        id: typing.Optional[str] = None,
        comment: typing.Optional[str] = None,
        observation_id: typing.Optional[str] = None,
        kwargs=None,
    ):
        """Create a score with the provided parameters.

        This function is used to create a score that can be associated with an observation or a trace.
        If a trace ID is provided, the score is linked to that trace. If not, it defaults to using
        the instance's trace ID (self.trace_id). The function also supports linking the score to an
        observation. If no observation ID is provided, the score is associated with a trace.

        Args:
            name (str): The name of the score.
            value (float): The numerical value of the score.
            trace_id (Optional[str]): The trace ID to associate with the score. Defaults to the instance's trace ID.
            id (Optional[str]): The unique identifier of the score. If not provided, a new UUID is generated.
            comment (Optional[str]): An optional comment associated with the score.
            observation_id (Optional[str]): The id of the observation associated with the score.
            kwargs (Optional[dict]): Additional keyword arguments that can be included in the score.

        Returns:
            StatefulClient: A stateful client representing either the associated observation (if observation_id is provided)
            or the trace (if observation_id is not provided).

        Example:
            ```python
                from langfuse import Langfuse

                langfuse = Langfuse()

                # Get id of created trace
                trace_id = handler.get_trace_id()

                # Add score, e.g. via the Python SDK
                langfuse = Langfuse()
                trace = langfuse.score(
                    trace_id=trace_id,
                    name="user-explicit-feedback",
                    value=1,
                    comment="I like how personalized the response is"
                )
            ```
        """
        try:
            new_id = str(uuid.uuid4()) if id is None else id

            new_dict = {
                "id": new_id,
                "trace_id": self.trace_id if trace_id is None else trace_id,
                "observation_id": observation_id,
                "name": name,
                "value": value,
                "comment": comment,
            }

            if kwargs is not None:
                new_dict.update(kwargs)

            self.log.debug(f"Creating score {new_dict}...")
            new_body = ScoreBody(**new_dict)

            event = {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "body": new_body.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)

            if observation_id is not None:
                return StatefulClient(
                    self.client,
                    observation_id,
                    StateType.OBSERVATION,
                    trace_id,
                    self.task_manager,
                )
            else:
                return StatefulClient(
                    self.client, new_id, StateType.TRACE, new_id, self.task_manager
                )

        except Exception as e:
            self.log.exception(e)

    def span(
        self,
        *,
        id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ):
        """Create a span with the provided parameters.

        A span represents durations of units of work in a trace.
        If no trace_id is provided, a new unique trace identifier is generated for the span.
        If a parent_observation_id is provided, the span is linked to this existing observation.

        Args:
            id (Optional[str]): Unique identifier of the span. If not provided, a new UUID is generated.
            trace_id (Optional[str]): The trace ID associated with the span. If not provided, a new UUID is generated.
            name (Optional[str]): The name of the span.
            start_time (Optional[datetime]): The start time of the span. Defaults to the current time if not provided.
            end_time (Optional[datetime]): The end time of the span.
            metadata (Optional[Any]): Additional metadata associated with the span.
            input (Optional[Any]): Input data of the span.
            output (Optional[Any]): Output data of the span.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize span.
            status_message (Optional[str]): A status message associated with the span.
            parent_observation_id (Optional[str]): The ID of the parent observation, if any.
            version (Optional[str]): Version of the span.
            **kwargs: Additional keyword arguments to include in the span.

        Returns:
            StatefulSpanClient: An stateful client representing the created span.

        Example:
            ```python
                from langfuse import Langfuse

                langfuse = Langfuse()

                trace = langfuse.trace(name = "llm-feature")
                retrieval = langfuse.span(name = "retrieval", trace_id = trace.id)
                retrieval.event(
                    name = "db-summary"
                )
            ```
        retrieval = trace.span(name = "retrieval")
        """
        try:
            new_span_id = str(uuid.uuid4()) if id is None else id
            new_trace_id = str(uuid.uuid4()) if trace_id is None else trace_id
            self.trace_id = new_trace_id

            span_body = {
                "id": new_span_id,
                "trace_id": new_trace_id,
                "name": name,
                "start_time": start_time
                if start_time is not None
                else _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "parent_observation_id": parent_observation_id,
                "version": version,
                "end_time": end_time,
                "trace": {"release": self.release},
            }
            if kwargs is not None:
                span_body.update(kwargs)

            if trace_id is None:
                self._generate_trace(new_trace_id, name)

            self.log.debug(f"Creating span {span_body}...")

            span_body = CreateSpanBody(**span_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "span-create",
                "body": span_body.dict(exclude_none=True),
            }

            self.log.debug(f"Creating span {event}...")
            self.task_manager.add_task(event)

            return StatefulSpanClient(
                self.client,
                new_span_id,
                StateType.OBSERVATION,
                new_trace_id,
                self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def event(
        self,
        *,
        id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ):
        """Create a event with the provided parameters.

        An event represents a discrete event in a trace.
        If no trace_id is provided, a new unique trace identifier is generated for the event.
        If a parent_observation_id is provided, the event is linked to this existing observation.

        Args:
            id (Optional[str]): Unique identifier for the event. If not provided, a new UUID is generated.
            trace_id (Optional[str]): The trace ID associated with this event. If not provided, a new UUID is generated.
            name (Optional[str]): The name of the event.
            start_time (Optional[datetime]): The start time of the event. Defaults to the current time if not provided.
            metadata (Optional[Any]): Additional metadata associated with the event.
            input (Optional[Any]): Input data of the event.
            output (Optional[Any]): Output data of the event.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize event.
            status_message (Optional[str]): A status message associated with the event.
            parent_observation_id (Optional[str]): The ID of the parent observation, if applicable.
            version (Optional[str]): Version information for the event.
            **kwargs: Additional keyword arguments to include in the event.

        Returns:
            StatefulSpanClient: An stateful client representing the created event.
        """
        try:
            event_id = str(uuid.uuid4()) if id is None else id

            new_trace_id = str(uuid.uuid4()) if trace_id is None else trace_id
            self.trace_id = new_trace_id

            event_body = {
                "id": event_id,
                "trace_id": new_trace_id,
                "name": name,
                "start_time": start_time
                if start_time is not None
                else _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "parent_observation_id": parent_observation_id,
                "version": version,
                "trace": {"release": self.release},
            }

            if kwargs is not None:
                event_body.update(kwargs)

            if trace_id is None:
                self._generate_trace(new_trace_id, name)

            request = CreateEventBody(**event_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "event-create",
                "body": request.dict(exclude_none=True),
            }

            self.log.debug(f"Creating event {event}...")
            self.task_manager.add_task(event)

            return StatefulSpanClient(
                self.client,
                event_id,
                StateType.OBSERVATION,
                new_trace_id,
                self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def generation(
        self,
        *,
        id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        usage: typing.Optional[typing.Union[pydantic.BaseModel, ModelUsage]] = None,
        prompt: typing.Optional[PromptClient] = None,
        **kwargs,
    ):
        """Create a generation with the provided parameters.

        A generation is a span which is used to log generations of AI models.
        If no trace_id is provided, a new unique trace identifier is generated for the generation record.
        If a parent_observation_id is provided, the generation is linked to this existing observation.

        Args:
            id (Optional[str]): Unique identifier for the generation. If not provided, a new UUID is generated.
            trace_id (Optional[str]): The trace ID associated with this generation.
            If not provided, a new UUID is generated and used.
            name (Optional[str]): The name of the generation.
            start_time (Optional[datetime]): The start time of the generation.
            Defaults to the current time if not provided.
            end_time (Optional[datetime]): The end time of the generation.
            metadata (Optional[Any]): Additional metadata associated with the generation.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize generation.
            status_message (Optional[str]): A status message associated with the generation.
            parent_observation_id (Optional[str]): The id of the parent observation, if applicable.
            version (Optional[str]): Version information of the generation.
            completion_start_time (Optional[datetime]): The time when the generation was completed.
            model (Optional[str]): The model used for the generation process.
            model_parameters (Optional[Dict[str, MapValue]]): Parameters of the model used for the generation.
            input (Optional[Any]): Input data of the generation.
            output (Optional[Any]): Output data of the generation.
            usage (Optional[Union[BaseModel, ModelUsage]]): Usage information on the generation.
            prompt (Optional[PromptClient]): Associated prompt template used for the generation.
            **kwargs: Additional keyword arguments to include in the generation.

        Returns:
            StatefulGenerationClient: An object representing the created generation, allowing for further interactions and state management.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a generation in Langfuse
            generation = langfuse.generation(
                name="summary-generation",
                model="gpt-3.5-turbo",
                model_parameters={"maxTokens": "1000", "temperature": "0.9"},
                input=[{"role": "system", "content": "You are a helpful assistant."}, 
                       {"role": "user", "content": "Please generate a summary of the following documents ..."}],
                metadata={"interface": "whatsapp"}
            )
            ```
        """
        try:
            new_trace_id = str(uuid.uuid4()) if trace_id is None else trace_id
            new_generation_id = str(uuid.uuid4()) if id is None else id
            self.trace_id = new_trace_id

            generation_body = {
                "id": new_generation_id,
                "trace_id": new_trace_id,
                "release": self.release,
                "name": name,
                "start_time": start_time
                if start_time is not None
                else _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "parent_observation_id": parent_observation_id,
                "version": version,
                "end_time": end_time,
                "completion_start_time": completion_start_time,
                "model": model,
                "model_parameters": model_parameters,
                "usage": _convert_usage_input(usage) if usage is not None else None,
                "trace": {"release": self.release},
                **_create_prompt_context(prompt),
            }
            if kwargs is not None:
                generation_body.update(kwargs)

            if trace_id is None:
                trace = {
                    "id": new_trace_id,
                    "release": self.release,
                    "name": name,
                }
                request = TraceBody(**trace)

                event = {
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "body": request.dict(exclude_none=True),
                }

                self.log.debug(f"Creating trace {event}...")

                self.task_manager.add_task(event)

            self.log.debug(f"Creating generation max {generation_body} {usage}...")
            request = CreateGenerationBody(**generation_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "body": request.dict(exclude_none=True),
            }

            self.log.debug(f"Creating top-level generation {event} ...")
            self.task_manager.add_task(event)

            return StatefulGenerationClient(
                self.client,
                new_generation_id,
                StateType.OBSERVATION,
                new_trace_id,
                self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def _generate_trace(self, trace_id: str, name: str):
        """Create a trace with the specified trace ID and name."""
        trace_dict = {
            "id": trace_id,
            "release": self.release,
            "name": name,
        }

        trace_body = TraceBody(**trace_dict)

        event = {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "body": trace_body.dict(exclude_none=True),
        }

        self.log.debug(f"Creating trace {event}...")
        self.task_manager.add_task(event)

    # On program exit, allow the consumer thread to exit cleanly.
    # This prevents exceptions and a messy shutdown when the
    # interpreter is destroyed before the daemon thread finishes
    # execution. However, it is *not* the same as flushing the queue!
    # To guarantee all messages have been delivered, you'll still need to call flush().
    def join(self):
        """End the consumer threads once the queue is empty and blocks execution until finished."""
        try:
            return self.task_manager.join()
        except Exception as e:
            self.log.exception(e)

    def flush(self):
        """Force a flush from the internal queue to the server.

        This method should be used every time to exit Langfuse cleanly to ensure all queued events are processed.
        This method waits for all events in the queue to be processed and sent to the server. It blocks
        until the queue is empty. The method logs the total number of items approximately flushed due
        to potential variations caused by threading.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Some operations with Langfuse

            # Flushing all events to end Langfuse cleanly
            langfuse.flush()
            ```
        """
        try:
            return self.task_manager.flush()
        except Exception as e:
            self.log.exception(e)

    def shutdown(self):
        """Initiate a graceful shutdown of the task manager, ensuring all tasks are flushed and processed."""
        try:
            return self.task_manager.shutdown()
        except Exception as e:
            self.log.exception(e)


class StateType(Enum):
    """Enumeration to distinguish observation and trace states.

    Attributes:
        OBSERVATION (int): Identifier for state type observation.
        TRACE (int): Identifier for state type trace.
    """
    OBSERVATION = 1
    TRACE = 0


class StatefulClient(object):
    """Base class for handling stateful operations in the Langfuse system.

    This client is capable of creating different Lagnfuse objects like spans, generations, scores, and events,
    associating them with either an observation or a trace based on the specified state type.

    Attributes:
        client (FernLangfuse): Core interface for Langfuse API interactions.
        id (str): Unique identifier of the stateful client (either observation or trace).
        state_type (StateType): Enum indicating whether the client is an observation or a trace.
        trace_id (str): Id of the trace associated with the stateful client.
        task_manager (TaskManager): Manager handling asynchronous tasks for the client.
    """
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        client: FernLangfuse,
        id: str,
        state_type: StateType,
        trace_id: str,
        task_manager: TaskManager,
    ):
        self.client = client
        self.trace_id = trace_id
        self.id = id
        self.state_type = state_type
        self.task_manager = task_manager

    def _add_state_to_event(self, body: dict):
        """Add state information to an event body based on the state type of the client.

        If the state type is an observation, it adds the parent observation ID and trace ID.
        If the state type is a trace, it adds only the trace ID.
        """
        if self.state_type == StateType.OBSERVATION:
            body["parent_observation_id"] = self.id
            body["trace_id"] = self.trace_id
        else:
            body["trace_id"] = self.id
        return body

    def _add_default_values(self, body: dict):
        """If not already set, add default values (start time) to the event body."""
        if body.get("start_time") is None:
            body["start_time"] = _get_timestamp()
        return body

    def generation(
        self,
        *,
        id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        usage: typing.Optional[typing.Union[pydantic.BaseModel, ModelUsage]] = None,
        prompt: typing.Optional[PromptClient] = None,
        **kwargs,
    ):
        """Create a generation.

        Constructs the body of the generation, adding default values and state information,
        and then queues it for asynchronous processing.

        Args:
            id (str, optional): Unique identifier for the generation. Defaults to None, which generates a new UUID.
            name (Optional[str]): The name of the generation.
            start_time (Optional[datetime]): The start time of the generation. Defaults to the current time if not provided.
            end_time (Optional[datetime]): The end time of the generation.
            metadata (Optional[Any]): Additional metadata associated with the generation.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize generation.
            status_message (Optional[str]): A status message associated with the generation.
            version (Optional[str]): Version information of the generation.
            completion_start_time (Optional[datetime]): The time when the generation was completed.
            model (Optional[str]): The model used for the generation process.
            model_parameters (Optional[Dict[str, MapValue]]): Parameters of the model used for the generation.
            input (Optional[Any]): Input data of the generation.
            output (Optional[Any]): Output data of the generation.
            usage (Optional[Union[BaseModel, ModelUsage]]): Usage information on the generation.
            prompt (Optional[PromptClient]): Prompt template used for the generation.
            **kwargs: Additional keyword arguments to include in the generation.

        Returns:
            StatefulGenerationClient: The created generation.

        Example: Using StatefulTraceClient to create a generation
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            trace = langfuse.trace(name="capital-guesser")
            generation = trace.generation(
                name="generate-text",
                model="gpt-3.5-turbo",
                input="What is the capital of France?",
                model_parameters={"temperature": 0.5}
            )
            ```
        """
        try:
            generation_id = str(uuid.uuid4()) if id is None else id

            generation_body = {
                "id": generation_id,
                "name": name,
                "start_time": start_time
                if start_time is not None
                else _get_timestamp(),
                "metadata": metadata,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time,
                "completion_start_time": completion_start_time,
                "model": model,
                "model_parameters": model_parameters,
                "input": input,
                "output": output,
                "usage": _convert_usage_input(usage) if usage is not None else None,
                **_create_prompt_context(prompt),
            }

            if kwargs is not None:
                generation_body.update(kwargs)

            generation_body = self._add_state_to_event(generation_body)
            new_body = self._add_default_values(generation_body)

            new_body = CreateGenerationBody(**new_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "body": new_body.dict(exclude_none=True),
            }

            self.log.debug(f"Creating generation {new_body}...")
            self.task_manager.add_task(event)

            return StatefulGenerationClient(
                self.client,
                generation_id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def span(
        self,
        *,
        id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ):
        """Create a span.

        Constructs the body of the span, adding default values and state information,
        and then queues it for asynchronous processing.

        Args:
            id (str, optional): Unique identifier for the span. Defaults to None, which generates a new UUID.
            name (Optional[str]): The name of the span.
            start_time (Optional[datetime]): The start time of the span. Defaults to the current time if not provided.
            end_time (Optional[datetime]): The end time of the span.
            metadata (Optional[Any]): Additional metadata associated with the span.
            input (Optional[Any]): Input data of the span.
            output (Optional[Any]): Output data of the span.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize span.
            status_message (Optional[str]): A status message associated with the span.
            version (Optional[str]): Version of the span.
            **kwargs: Additional keyword arguments to include in the span.

        Returns:
            StatefulSpanClient: The created span.

        Example: Using StatefulTraceClient to create a span
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            trace = langfuse.trace(name="RAG")
            span = trace.span(
                name="embedding-search",
                metadata={"database": "example-database"},
                input = {'query': 'This document entails ...'},
            )
            ```
        """
        try:
            span_id = str(uuid.uuid4()) if id is None else id

            span_body = {
                "id": span_id,
                "name": name,
                "start_time": start_time
                if start_time is not None
                else _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time,
            }

            if kwargs is not None:
                span_body.update(kwargs)

            self.log.debug(f"Creating span {span_body}...")

            new_dict = self._add_state_to_event(span_body)
            new_body = self._add_default_values(new_dict)

            event = CreateSpanBody(**new_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "span-create",
                "body": event.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)
            return StatefulSpanClient(
                self.client,
                span_id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def score(
        self,
        *,
        id: typing.Optional[str] = None,
        name: str,
        value: float,
        comment: typing.Optional[str] = None,
        kwargs=None,
    ):
        """Create a score.

        Constructs the body of the score, adding state information,
        and then queues it for asynchronous processing.

        Args:
            id (str, optional): Unique identifier for the score. Defaults to None, which generates a new UUID.
            name (str): The name of the score.
            value (float): The numerical value of the score.
            comment (Optional[str]): An optional comment associated with the score.
            kwargs (Optional[dict]): Additional keyword arguments that can be included in the score.

        Returns:
            StatefulClient: The created score.

        Example: Using StatefulSpanClient to create a score
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()
            
            trace = langfuse.trace(name="example-trace")
            trace.score(
                name="user-explicit-feedback",
                value=1,
                comment="I like this example."
            )
            ```
        """
        try:
            score_id = str(uuid.uuid4()) if id is None else id

            new_score = {
                "id": score_id,
                "trace_id": self.trace_id,
                "name": name,
                "value": value,
                "comment": comment,
            }

            if kwargs is not None:
                new_score.update(kwargs)

            self.log.debug(f"Creating score {new_score}...")

            new_dict = self._add_state_to_event(new_score)

            if self.state_type == StateType.OBSERVATION:
                new_dict["observationId"] = self.id

            request = ScoreBody(**new_dict)

            event = {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "body": request.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)
            return StatefulClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def event(
        self,
        *,
        id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ):
        """Create an event.

        Constructs the body of the event, adding default values and state information,
        and then queues it for asynchronous processing.

        Args:
            id (str, optional): Unique identifier for the event. Defaults to None, which generates a new UUID.
            name (Optional[str]): The name of the event.
            start_time (Optional[datetime]): The start time of the event. Defaults to the current time if not provided.
            metadata (Optional[Any]): Additional metadata associated with the event.
            input (Optional[Any]): Input data of the event.
            output (Optional[Any]): Output data of the event.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize the event.
            status_message (Optional[str]): A status message associated with the event.
            version (Optional[str]): Version information for the event.
            **kwargs: Additional keyword arguments to include in the event.

        Returns:
            StatefulClient: The created event.

        Example: (Span)
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            trace = langfuse.trace(name="example-trace")
            span = trace.span(name="example-span")
            span.event(
                name="example-event",
                input={"text": "This is an event."},
            )
            ```
        """
        try:
            event_id = str(uuid.uuid4()) if id is None else id

            event_body = {
                "id": event_id,
                "name": name,
                "start_time": start_time
                if start_time is not None
                else _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "version": version,
            }

            if kwargs is not None:
                event_body.update(kwargs)

            new_dict = self._add_state_to_event(event_body)
            new_body = self._add_default_values(new_dict)

            request = CreateEventBody(**new_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "event-create",
                "body": request.dict(exclude_none=True),
            }

            self.log.debug(f"Creating event {event}...")
            self.task_manager.add_task(event)

            return StatefulClient(
                self.client, event_id, self.state_type, self.trace_id, self.task_manager
            )
        except Exception as e:
            self.log.exception(e)

    def get_trace_url(self):
        """Get the URL to see the current trace in the Langfuse UI."""
        return f"{self.client._client_wrapper._base_url}/trace/{self.trace_id}"


class StatefulGenerationClient(StatefulClient):
    """Class for handling stateful operations of generations in the Langfuse system. Inherits from StatefulClient.

    This client extends the capabilities of the StatefulClient to specifically handle generation,
    allowing for the creation, update, and termination of generation processes in Langfuse.

    Attributes:
        client (FernLangfuse): Core interface for Langfuse API interaction.
        id (str): Unique identifier of the generation.
        state_type (StateType): Type of the stateful entity (observation or trace).
        trace_id (str): Id of trace associated with the generation.
        task_manager (TaskManager): Manager for handling asynchronous tasks.
    """
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        client: FernLangfuse,
        id: str,
        state_type: StateType,
        trace_id: str,
        task_manager: TaskManager,
    ):
        super().__init__(client, id, state_type, trace_id, task_manager)

    def update(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        usage: typing.Optional[typing.Union[pydantic.BaseModel, ModelUsage]] = None,
        prompt: typing.Optional[PromptClient] = None,
        **kwargs,
    ):
        """Update the properties of an existing generation and schedules the updated generation for asynchronous processing.

        This method allows for the modification of various attributes related to a generation. The changes 
        are serialized and scheduled for processing. Attributes that can be updated include the generation's name, 
        start and end times, metadata, model details, and prompt information (template name and version).

        Args:
        name (Optional[str]): The name of the generation to update.
        start_time (Optional[dt.datetime]): The start time for the generation.
        end_time (Optional[dt.datetime]): The end time for the generation.
        metadata (Optional[typing.Any]): Additional metadata for the generation.
        level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize the generation.
        status_message (Optional[str]): A status message for the generation.
        version (Optional[str]): The version of the generation.
        completion_start_time (Optional[dt.datetime]): Start time of the completion phase of the generation.
        model (Optional[str]): The AI model used in the generation.
        model_parameters (Optional[typing.Dict[str, MapValue]]): Parameters for the model used in the generation.
        input (Optional[typing.Any]): Input data for the generation.
        output (Optional[typing.Any]): Output data from the generation.
        usage (Optional[typing.Union[pydantic.BaseModel, ModelUsage]]): Usage information for the generation.
        prompt (Optional[PromptClient]): The prompt template used in the generation.
        **kwargs: Additional keyword arguments for custom parameters.

        Returns:
            StatefulGenerationClient: The updated generation.
        """
        try:
            generation_body = {
                "id": self.id,
                "name": name,
                "start_time": start_time,
                "metadata": metadata,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time,
                "completion_start_time": completion_start_time,
                "model": model,
                "model_parameters": model_parameters,
                "input": input,
                "output": output,
                "usage": _convert_usage_input(usage) if usage is not None else None,
                **_create_prompt_context(prompt),
            }

            if kwargs is not None:
                generation_body.update(kwargs)
            self.log.debug(f"Update generation {generation_body}...")

            request = UpdateGenerationBody(**generation_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "generation-update",
                "body": request.dict(exclude_none=True),
            }

            self.log.debug(f"Update generation {event}...")
            self.task_manager.add_task(event)

            return StatefulGenerationClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def end(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        prompt_tokens: typing.Optional[int] = None,
        completion_tokens: typing.Optional[int] = None,
        total_tokens: typing.Optional[int] = None,
        **kwargs,
    ):
        """Conclude a generation by marking its end time and updating all relevant properties.

        This method is designed to finalize a generation. It sets the end time of the 
        generation to the current time if not explicitly provided and updates other attributes as needed. 
        The method essentially acts as a wrapper around the `update` method, providing a convenient way to 
        signal the completion of a generation.

        Args:
            name (Optional[str]): The name of the generation to conclude.
            start_time (Optional[dt.datetime]): The start time of the generation.
            end_time (Optional[dt.datetime]): The end time of the generation, defaults to current time.
            metadata (Optional[typing.Any]): Additional metadata for the generation.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level of the generation.
            status_message (Optional[str]): A status message for the generation.
            version (Optional[str]): The version of the generation.
            completion_start_time (Optional[dt.datetime]): Start time of the completion phase of the generation.
            model (Optional[str]): The AI model used in the generation.
            model_parameters (Optional[typing.Dict[str, MapValue]]): Parameters for the model used in the generation.
            input (Optional[typing.Any]): Input data for the generation.
            output (Optional[typing.Any]): Output data from the generation.
            prompt_tokens (Optional[int]): Number of tokens used in the prompt.
            completion_tokens (Optional[int]): Number of tokens generated in the completion phase.
            total_tokens (Optional[int]): Total number of tokens used in the generation.
            **kwargs: Additional keyword arguments for custom parameters.

        Returns:
            StatefulGenerationClient: An instance representing the concluded generation.
        """
        try:
            generation_body = {
                "name": name,
                "start_time": start_time,
                "metadata": metadata,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time if end_time is not None else _get_timestamp(),
                "completion_start_time": completion_start_time,
                "model": model,
                "model_parameters": model_parameters,
                "input": input,
                "output": output,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            if kwargs is not None:
                generation_body.update(kwargs)

            return self.update(**generation_body)

        except Exception as e:
            self.log.warning(e)


class StatefulSpanClient(StatefulClient):
    """Class for handling stateful operations of spans in the Langfuse system. Inherits from StatefulClient.

    This client extends the functionality of the StatefulClient to specifically handle spans,
    allowing for creating, updating, and concluding span processes in the Langfuse environment.

    Attributes:
        client (FernLangfuse): Core interface for Langfuse API interaction.
        id (str): Unique identifier of the span.
        state_type (StateType): Type of the stateful entity (observation or trace).
        trace_id (str): Id of trace associated with the span.
        task_manager (TaskManager): Manager for handling asynchronous tasks.
    """
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        client: FernLangfuse,
        id: str,
        state_type: StateType,
        trace_id: str,
        task_manager: TaskManager,
    ):
        super().__init__(client, id, state_type, trace_id, task_manager)

    def update(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ):
        """Update the properties of an existing generation and schedules the updated generation for asynchronous processing.

        This method allows modification of various attributes of a span, such as name, time frames, metadata,
        input, output, and more. The changes are serialized and scheduled for processing.

        Args:
            name (Optional[str]): The name of the span.
            start_time (Optional[datetime]): The start time of the span.
            end_time (Optional[datetime]): The end time of the span.
            metadata (Optional[Any]): Additional metadata for the span.
            input (Optional[Any]): Input data of the span.
            output (Optional[Any]): Output data of the span.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize span.
            status_message (Optional[str]): Status message associated with the span.
            version (Optional[str]): Version of the span.
            **kwargs: Additional keyword arguments for custom parameters.

        Returns:
            StatefulSpanClient: An updated span.
        """
        try:
            span_body = {
                "id": self.id,
                "name": name,
                "start_time": start_time,
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time,
            }
            if kwargs is not None:
                span_body.update(kwargs)
            self.log.debug(f"Update span {span_body}...")

            request = UpdateSpanBody(**span_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "span-update",
                "body": request.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)
            return StatefulSpanClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def end(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ):
        """Conclude a span by marking its end time and updating all relevant properties.

        This method is designed to finalize a span. It sets the end time of the 
        span to the current time if not explicitly provided and updates other attributes as needed. 
        The method essentially acts as a wrapper around the `update` method, providing a convenient way to 
        signal the completion of a span.

        Args:
            name (Optional[str]): The name of the span.
            start_time (Optional[datetime]): The start time of the span.
            end_time (Optional[datetime]): The end time of the span. Defaults to current time if not provided.
            metadata (Optional[Any]): Additional metadata for the span.
            input (Optional[Any]): Input data of the span.
            output (Optional[Any]): Output data of the span.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): Logging level to categorize span.
            status_message (Optional[str]): Status message associated with the span.
            version (Optional[str]): Version of the span.
            **kwargs: Additional keyword arguments for custom parameters.

        Returns:
            StatefulSpanClient: The concluded span.
        """
        try:
            span_body = {
                "name": name,
                "start_time": start_time,
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time if end_time is not None else _get_timestamp(),
            }
            if kwargs is not None:
                span_body.update(kwargs)

            return self.update(**span_body)

        except Exception as e:
            self.log.warning(e)

    def get_langchain_handler(self):
        """Get langchain callback handler associated with the current span.

        Returns:
            CallbackHandler: An instance of CallbackHandler linked to this StatefulSpanClient.
        """
        from langfuse.callback import CallbackHandler

        return CallbackHandler(stateful_client=self)


class StatefulTraceClient(StatefulClient):
    """Class for handling stateful operations of traces in the Langfuse system. Inherits from StatefulClient.

    This client extends the StatefulClient's capabilities to handle traces, enabling the creation and
    updating of trace processes in the Langfuse environment.

    Attributes:
        client (FernLangfuse): Core interface for Langfuse API interaction.
        id (str): Unique identifier of the trace.
        state_type (StateType): Type of the stateful entity (observation or trace).
        trace_id (str): The trace ID associated with this client.
        task_manager (TaskManager): Manager for handling asynchronous tasks.
    """
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        client: FernLangfuse,
        id: str,
        state_type: StateType,
        trace_id: str,
        task_manager: TaskManager,
    ):
        super().__init__(client, id, state_type, trace_id, task_manager)
        self.task_manager = task_manager

    def update(
        self,
        *,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        metadata: typing.Optional[typing.Any] = None,
        tags: typing.Optional[typing.List[str]] = None,
        **kwargs,
    ):
        """Update the properties of an existing trace and schedules the updated trace for asynchronous processing.

        This method allows for the modification of various attributes of a trace, such as name, user ID,
        version, input, output, metadata, and tags. The updated trace is queued in the task manager for later execution.

        Args:
            name (Optional[str]): The name of the trace.
            user_id (Optional[str]): The user ID associated with the trace.
            version (Optional[str]): The version of the trace.
            input (Optional[Any]): Input data of the trace.
            output (Optional[Any]): Output data of the trace.
            metadata (Optional[Any]): Additional metadata for the trace.
            tags (Optional[List[str]]): Tags associated with the trace.
            **kwargs: Additional keyword arguments for custom parameters.

        Returns:
            StatefulTraceClient: The updated trace.
        """
        try:
            trace_body = {
                "id": self.id,
                "name": name,
                "userId": user_id,
                "version": version,
                "input": input,
                "output": output,
                "metadata": metadata,
                "tags": tags,
            }
            if kwargs is not None:
                trace_body.update(kwargs)
            self.log.debug(f"Update trace {trace_body}...")

            request = TraceBody(**trace_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "body": request.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)
            return StatefulTraceClient(
                self.client,
                self.id,
                StateType.TRACE,
                self.trace_id,
                task_manager=self.task_manager,
            )
        except Exception as e:
            self.log.exception(e)

    def get_langchain_handler(self):
        """Get langchain callback handler associated with the current trace.

        This method creates and returns a CallbackHandler instance, linking it with the current
        trace, enabling automatic tracing in Langfuse

        Raises:
            ImportError: If the 'langchain' module is not installed, indicating missing functionality.

        Returns:
            CallbackHandler: An instance of CallbackHandler linked to this trace.
        """
        try:
            # adding this to ensure langchain is installed
            import langchain  # noqa

            from langfuse.callback import CallbackHandler

            self.log.debug(f"Creating new handler for trace {self.id}")

            return CallbackHandler(
                stateful_client=self, debug=self.log.level == logging.DEBUG
            )
        except ImportError as e:
            self.log.exception(
                f"Could not import langchain. Some functionality may be missing. {e.message}"
            )

        except Exception as e:
            self.log.exception(e)

    def getNewHandler(self):
        """Alias for the `get_langchain_handler` method. Retrieves a callback handler for the trace."""
        return self.get_langchain_handler()


class DatasetItemClient:
    """Class for managing dataset items in Langfuse.

    Args:
        id (str): Unique identifier of the dataset item.
        status (DatasetStatus): The status of the dataset item. Can be either 'ACTIVE' or 'ARCHIVED'.
        input (Any): Input data associated of the dataset item.
        expected_output (Optional[Any]): Expected output of the dataset item.
        source_observation_id (Optional[str]): Identifier of the source observation.
        dataset_id (str): Identifier of the dataset to which this item belongs.
        created_at (datetime): Timestamp of dataset item creation.
        updated_at (datetime): Timestamp of the last update to the dataset item.
        langfuse (Langfuse): Instance of Langfuse client for API interactions.
    
    Example: Print the input of each dataset item in a dataset.
        ```python
        from langfuse import Langfuse

        langfuse = Langfuse()

        dataset = langfuse.get_dataset("<dataset_name>")
 
        for item in dataset.items:
            # Generate a completion using the input of every item
            completion, generation = llm_app.run(item.input)

            # Evaluate the completion
            generation.score(
                name="example-score",
                value=1
            )
        ```
    """
    id: str
    status: DatasetStatus
    input: typing.Any
    expected_output: typing.Optional[typing.Any]
    source_observation_id: typing.Optional[str]
    dataset_id: str
    created_at: dt.datetime
    updated_at: dt.datetime

    langfuse: Langfuse

    def __init__(self, dataset_item: DatasetItem, langfuse: Langfuse):
        self.id = dataset_item.id
        self.status = dataset_item.status
        self.input = dataset_item.input
        self.expected_output = dataset_item.expected_output
        self.source_observation_id = dataset_item.source_observation_id
        self.dataset_id = dataset_item.dataset_id
        self.created_at = dataset_item.created_at
        self.updated_at = dataset_item.updated_at

        self.langfuse = langfuse

    def flush(self, observation: StatefulClient, run_name: str):
        """Flushes an observations task manager's queue.
        
        Used before creating a dataset run item to ensure all events are persistent.

        Args:
            observation (StatefulClient): The observation client associated with the dataset item.
            run_name (str): The name of the dataset run.
        """
        observation.task_manager.flush()

    def link(self, observation: typing.Union[StatefulClient, str], run_name: str):
        """Link the dataset item to an observation within a specific dataset run.

        Flushes the observations's task manager queue before creating the dataset run item.

        Args:
            observation (Union[StatefulClient, str]): The observation to link, either as a client or as an ID.
            run_name (str): The name of the dataset run to which the item is linked.
        """
        observation_id = None

        if isinstance(observation, StatefulClient):
            # flush the queue before creating the dataset run item
            # to ensure that all events are persisted.
            observation.task_manager.flush()
            observation_id = observation.id
        elif isinstance(observation, str):
            self.langfuse.flush()
            observation_id = observation
        else:
            raise ValueError(
                "observation parameter must be either a StatefulClient or a string"
            )

        logging.debug(
            f"Creating dataset run item: {run_name} {self.id} {observation_id}"
        )
        self.langfuse.client.dataset_run_items.create(
            request=CreateDatasetRunItemRequest(
                runName=run_name, datasetItemId=self.id, observationId=observation_id
            )
        )

    def get_langchain_handler(self, *, run_name: str):
        """Create and get a langchain callback handler linked to this dataset item.

        Creates a trace and a span, linked to the trace, and returns a Langchain CallbackHandler to the span.

        Args:
            run_name (str): The name of the dataset run to be used in the callback handler.

        Returns:
            CallbackHandler: An instance of CallbackHandler linked to the created span.
        """
        from langfuse.callback import CallbackHandler

        metadata = {
            "dataset_item_id": self.id,
            "run_name": run_name,
            "dataset_id": self.dataset_id,
        }
        trace = self.langfuse.trace(name="dataset-run", metadata=metadata)
        span = trace.span(name="dataset-run", metadata=metadata)

        self.langfuse.flush()

        self.link(span, run_name)

        return CallbackHandler(stateful_client=span)

class DatasetClient:
    """Class for managing datasets in Langfuse.

    Attributes:
        id (str): Unique identifier of the dataset.
        name (str): Name of the dataset.
        project_id (str): Identifier of the project to which the dataset belongs.
        dataset_name (str): Name of the dataset.
        created_at (datetime): Timestamp of dataset creation.
        updated_at (datetime): Timestamp of the last update to the dataset.
        items (List[DatasetItemClient]): List of dataset items associated with the dataset.
        runs (List[str]): List of dataset runs associated with the dataset.

    Example: Print the input of each dataset item in a dataset.
        ```python
        from langfuse import Langfuse

        langfuse = Langfuse()

        dataset = langfuse.get_dataset("<dataset_name>")
 
        for item in dataset.items:
            print(item.input)
        ```
    """
    id: str
    name: str
    project_id: str
    dataset_name: str
    created_at: dt.datetime
    updated_at: dt.datetime
    items: typing.List[DatasetItemClient]
    runs: typing.List[str]

    def __init__(self, dataset: Dataset, items: typing.List[DatasetItemClient]):
        self.id = dataset.id
        self.name = dataset.name
        self.project_id = dataset.project_id
        self.dataset_name = dataset.name
        self.created_at = dataset.created_at
        self.updated_at = dataset.updated_at
        self.items = items
        self.runs = dataset.runs