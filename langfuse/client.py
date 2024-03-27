import datetime as dt
import logging
import os
import typing
import uuid
import httpx
from enum import Enum
from typing import Any, Optional

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
from langfuse.api.resources.observations.types.observations_views import (
    ObservationsViews,
)
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
from langfuse.types import SpanLevel
from langfuse.utils import _convert_usage_input, _create_prompt_context, _get_timestamp

from .version import __version__ as version


class Langfuse(object):
    """Langfuse Python client.

    Attributes:
        log (logging.Logger): Logger for the Langfuse client.
        base_url (str): Base URL of the Langfuse API, serving as the root address for API endpoint construction.
        httpx_client (httpx.Client): HTTPX client utilized for executing requests to the Langfuse API.
        client (FernLangfuse): Core interface for Langfuse API interaction.
        task_manager (TaskManager): Task Manager dedicated to handling asynchronous tasks.
        release (str): Identifies the release number or hash of the application.
        prompt_cache (PromptCache): A cache for efficiently storing and retrieving PromptClient instances.

    Example:
        Initiating the Langfuse client should always be first step to use Langfuse.
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
    """Logger for the Langfuse client."""

    host: str
    """Host of Langfuse API."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: bool = False,
        threads: int = 1,
        flush_at: int = 15,
        flush_interval: float = 0.5,
        max_retries: int = 3,
        timeout: int = 10,  # seconds
        sdk_integration: Optional[str] = "default",
        httpx_client: Optional[httpx.Client] = None,
    ):
        """Initialize the Langfuse client.

        Args:
            public_key: Public API key of Langfuse project. Can be set via `LANGFUSE_PUBLIC_KEY` environment variable.
            secret_key: Secret API key of Langfuse project. Can be set via `LANGFUSE_SECRET_KEY` environment variable.
            host: Host of Langfuse API. Can be set via `LANGFUSE_HOST` environment variable. Defaults to `https://cloud.langfuse.com`.
            release: Release number/hash of the application to provide analytics grouped by release. Can be set via `LANGFUSE_RELEASE` environment variable.
            debug: Enables debug mode for more verbose logging. Can be set via `LANGFUSE_DEBUG` environment variable.
            threads: Number of consumer threads to execute network requests. Helps scaling the SDK for high load. Only increase this if you run into scaling issues.
            flush_at: Max batch size that's sent to the API.
            flush_interval: Max delay until a new batch is sent to the API.
            max_retries: Max number of retries in case of API/network errors.
            timeout: Timeout of API requests in seconds.
            httpx_client: Pass your own httpx client for more customizability of requests.
            sdk_integration: Used by intgerations that wrap the Langfuse SDK to add context for debugging and support. Not to be used directly.

        Raises:
            ValueError: If public_key or secret_key are not set and not found in environment variables.

        Example:
            Initiating the Langfuse client should always be first step to use Langfuse.
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

        public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self.base_url = (
            host
            if host
            else os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

        if not public_key:
            self.log.warning("public_key is not set.")
            raise ValueError(
                "public_key is required, set as a parameter or environment variable 'LANGFUSE_PUBLIC_KEY'"
            )

        if not secret_key:
            self.log.warning("secret_key is not set.")
            raise ValueError(
                "secret_key is required, set as parameter or environment variable 'LANGFUSE_SECRET_KEY'"
            )

        self.httpx_client = httpx_client or httpx.Client(timeout=timeout)

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

        self.task_manager = TaskManager(**args)

        self.trace_id = None

        self.release = self._get_release_value(release)

        self.prompt_cache = PromptCache()

    def _get_release_value(self, release: Optional[str] = None) -> Optional[str]:
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
        """Get the URL of the current trace to view it in the Langfuse UI."""
        return f"{self.base_url}/trace/{self.trace_id}"

    def get_dataset(self, name: str) -> "DatasetClient":
        """Fetch a dataset by its name.

        Args:
            name (str): The name of the dataset to fetch.

        Returns:
            DatasetClient: The dataset with the given name.
        """
        try:
            self.log.debug(f"Getting datasets {name}")
            dataset = self.client.datasets.get(dataset_name=name)

            items = [DatasetItemClient(i, langfuse=self) for i in dataset.items]

            return DatasetClient(dataset, items=items)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_dataset_item(self, id: str) -> "DatasetItemClient":
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

        Note:
            This method is blocking. It is discouraged to use it in production code.
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
        """Get a dataset run.

        Args:
            dataset_name: Name of the dataset.
            dataset_run_name: Name of the dataset run.

        Returns:
            DatasetRun: The dataset run.
        """
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
        """Create a dataset with the given name on Langfuse.

        Args:
            name: Name of the dataset to create.

        Returns:
            Dataset: The created dataset as returned by the Langfuse API.
        """
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
        input: Any,
        expected_output: Optional[Any] = None,
        id: Optional[str] = None,
    ) -> DatasetItem:
        """Create a dataset item.

        Upserts if an item with id already exists.

        Args:
            dataset_name: Name of the dataset in which the dataset item should be created.
            input: Input data. Can contain any dict, list or scalar.
            expected_output: Expected output data. Defaults to None. Can contain any dict, list or scalar.
            id: Id of the dataset item. Defaults to None.

        Returns:
            DatasetItem: The created dataset item as returned by the Langfuse API.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Uploading items to the Langfuse dataset named "capital_cities"
            langfuse.create_dataset_item(
                dataset_name="capital_cities",
                input={"input": {"country": "Italy"}},
                expected_output={"expected_output": "Rome"}
            )
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
        """Get a trace via the Langfuse API by its id.

        Args:
            id: The id of the trace to fetch.

        Returns:
            TraceWithFullDetails: The trace with full details as returned by the Langfuse API.

        Raises:
            Exception: If the trace with the given id could not be found within the authenticated project or if an error occurred during the request.
        """
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
    ) -> ObservationsViews:
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

        Raises:
            Exception: If an error occurred during the request.
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
    ) -> ObservationsViews:
        """Get a list of generations in the current project matching the given parameters.

        Args:
            page (Optional[int]): Page number of the generations to return. Defaults to None.
            limit (Optional[int]): Maximum number of generations to return. Defaults to None.
            name (Optional[str]): Name of the generations to return. Defaults to None.
            user_id (Optional[str]): User identifier of the generations to return. Defaults to None.
            trace_id (Optional[str]): Trace identifier of the generations to return. Defaults to None.
            parent_observation_id (Optional[str]): Parent observation identifier of the generations to return. Defaults to None.

        Returns:
            List of ObservationsViews: List of generations in the project matching the given parameters.

        Raises:
            Exception: If an error occurred during the request.
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
        """Get an observation in the current project with the given identifier.

        Args:
            id: The identifier of the observation to fetch.

        Raises:
            Exception: If the observation with the given id could not be found within the authenticated project or if an error occurred during the request.
        """
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
        """Get a prompt.

        This method attempts to fetch the requested prompt from the local cache. If the prompt is not found
        in the cache or if the cached prompt has expired, it will try to fetch the prompt from the server again
        and update the cache. If fetching the new prompt fails, and there is an expired prompt in the cache, it will
        return the expired prompt as a fallback.

        Args:
            name (str): The name of the prompt to retrieve.
            version (Optional[int]): The version of the prompt. If not specified, the `active` version is returned.
            cache_ttl_seconds: Optional[int]: Time-to-live in seconds for caching the prompt. Must be specified as a
            keyword argument. If not set, defaults to 60 seconds.

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

    def create_prompt(
        self, *, name: str, prompt: str, is_active: bool, config: Optional[Any] = None
    ) -> PromptClient:
        """Create a new prompt in Langfuse.

        Args:
            name : The name of the prompt to be created.
            prompt : The content of the prompt to be created.
            is_active : A flag indicating whether the prompt is active or not.
            config: Additional structured data to be saved with the prompt. Defaults to None.

        Returns:
            PromptClient: The prompt.
        """
        try:
            self.log.debug(f"Creating prompt {name}, version {version}")

            if config is None:
                config = {}

            request = CreatePromptRequest(
                name=name,
                prompt=prompt,
                isActive=is_active,
                config=config,
            )
            server_prompt = self.client.prompts.create(request=request)
            return PromptClient(prompt=server_prompt)
        except Exception as e:
            self.log.exception(e)
            raise e

    def trace(
        self,
        *,
        id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        metadata: typing.Optional[typing.Any] = None,
        tags: typing.Optional[typing.List[str]] = None,
        timestamp: typing.Optional[dt.datetime] = None,
        public: typing.Optional[bool] = None,
        **kwargs,
    ) -> "StatefulTraceClient":
        """Create a trace.

        Args:
            id: The id of the trace can be set, defaults to a random id. Set it to link traces to external systems or when creating a distributed trace. Traces are upserted on id.
            name: Identifier of the trace. Useful for sorting/filtering in the UI.
            input: The input of the trace. Can be any JSON object.
            output: The output of the trace. Can be any JSON object.
            metadata: Additional metadata of the trace. Can be any JSON object. Metadata is merged when being updated via the API.
            user_id: The id of the user that triggered the execution. Used to provide user-level analytics.
            session_id: Used to group multiple traces into a session in Langfuse. Use your own session/thread identifier.
            version: The version of the trace type. Used to understand how changes to the trace type affect metrics. Useful in debugging.
            release: The release identifier of the current deployment. Used to understand how changes of different deployments affect metrics. Useful in debugging.
            tags: Tags are used to categorize or label traces. Traces can be filtered by tags in the UI and GET API. Tags can also be changed in the UI. Tags are merged and never deleted via the API.
            timestamp: The timestamp of the trace. Defaults to the current time if not provided.
            public: You can make a trace `public` to share it via a public link. This allows others to view the trace without needing to log in or be members of your Langfuse project.
            **kwargs: Additional keyword arguments that can be included in the trace.

        Returns:
            StatefulTraceClient: The created trace.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            trace = langfuse.trace(
                name="example-application",
                user_id="user-1234")
            )
            ```
        """
        new_id = id or str(uuid.uuid4())
        self.trace_id = new_id
        try:
            new_dict = {
                "id": new_id,
                "name": name,
                "userId": user_id,
                "sessionId": session_id
                or kwargs.get("sessionId", None),  # backward compatibility
                "release": self.release,
                "version": version,
                "metadata": metadata,
                "input": input,
                "output": output,
                "tags": tags,
                "timestamp": timestamp or _get_timestamp(),
                "public": public,
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

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulTraceClient(
                self.client, new_id, StateType.TRACE, new_id, self.task_manager
            )

    def score(
        self,
        *,
        name: str,
        value: float,
        trace_id: typing.Optional[str] = None,
        id: typing.Optional[str] = None,
        comment: typing.Optional[str] = None,
        observation_id: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulClient":
        """Create a score attached to a trace (and optionally an observation).

        Args:
            name (str): Identifier of the score.
            value (float): The value of the score. Can be any number, often standardized to 0..1
            trace_id (str): The id of the trace to which the score should be attached.
            comment (Optional[str]): Additional context/explanation of the score.
            observation_id (Optional[str]): The id of the observation to which the score should be attached.
            id (Optional[str]): The id of the score. If not provided, a new UUID is generated.
            **kwargs: Additional keyword arguments to include in the score.

        Returns:
            StatefulClient: Either the associated observation (if observation_id is provided) or the trace (if observation_id is not provided).

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name="example-application")

            # Get id of created trace
            trace_id = trace.id

            # Add score to the trace
            trace = langfuse.score(
                trace_id=trace_id,
                name="user-explicit-feedback",
                value=1,
                comment="I like how personalized the response is"
            )
            ```
        """
        trace_id = trace_id or self.trace_id or str(uuid.uuid4())
        new_id = id or str(uuid.uuid4())
        try:
            new_dict = {
                "id": new_id,
                "trace_id": trace_id,
                "observation_id": observation_id,
                "name": name,
                "value": value,
                "comment": comment,
                **kwargs,
            }

            self.log.debug(f"Creating score {new_dict}...")
            new_body = ScoreBody(**new_dict)

            event = {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "body": new_body.dict(exclude_none=True),
            }
            self.task_manager.add_task(event)

        except Exception as e:
            self.log.exception(e)
        finally:
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

    def span(
        self,
        *,
        id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulSpanClient":
        """Create a span.

        A span represents durations of units of work in a trace.
        Usually, you want to add a span nested within a trace. Optionally you can nest it within another observation by providing a parent_observation_id.

        If no trace_id is provided, a new trace is created just for this span.

        Args:
            id (Optional[str]): The id of the span can be set, otherwise a random id is generated. Spans are upserted on id.
            trace_id (Optional[str]): The trace ID associated with this span. If not provided, a new UUID is generated.
            parent_observation_id (Optional[str]): The ID of the parent observation, if applicable.
            name (Optional[str]): Identifier of the span. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime]): The time at which the span started, defaults to the current time.
            end_time (Optional[datetime]): The time at which the span ended. Automatically set by `span.end()`.
            metadata (Optional[dict]): Additional metadata of the span. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): The level of the span. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the span. Additional field for context of the event. E.g. the error message of an error event.
            input (Optional[dict]): The input to the span. Can be any JSON object.
            output (Optional[dict]): The output to the span. Can be any JSON object.
            version (Optional[str]): The version of the span type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            **kwargs: Additional keyword arguments to include in the span.

        Returns:
            StatefulSpanClient: The created span.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            trace = langfuse.trace(name = "llm-feature")

            # Create a span
            retrieval = langfuse.span(name = "retrieval", trace_id = trace.id)

            # Create a nested span
            nested_span = langfuse.span(name = "retrieval", trace_id = trace.id, parent_observation_id = retrieval.id)
            ```
        """
        new_span_id = id or str(uuid.uuid4())
        new_trace_id = trace_id or str(uuid.uuid4())
        self.trace_id = new_trace_id
        try:
            span_body = {
                "id": new_span_id,
                "trace_id": new_trace_id,
                "name": name,
                "start_time": start_time or _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "parent_observation_id": parent_observation_id,
                "version": version,
                "end_time": end_time,
                "trace": {"release": self.release},
                **kwargs,
            }

            if trace_id is None:
                self._generate_trace(new_trace_id, name or new_trace_id)

            self.log.debug(f"Creating span {span_body}...")

            span_body = CreateSpanBody(**span_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "span-create",
                "body": span_body.dict(exclude_none=True),
            }

            self.log.debug(f"Creating span {event}...")
            self.task_manager.add_task(event)

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulSpanClient(
                self.client,
                new_span_id,
                StateType.OBSERVATION,
                new_trace_id,
                self.task_manager,
            )

    def event(
        self,
        *,
        id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulSpanClient":
        """Create an event.

        An event represents a discrete event in a trace.
        Usually, you want to add a event nested within a trace. Optionally you can nest it within another observation by providing a parent_observation_id.

        If no trace_id is provided, a new trace is created just for this event.

        Args:
            id (Optional[str]): The id of the event can be set, otherwise a random id is generated.
            trace_id (Optional[str]): The trace ID associated with this event. If not provided, a new trace is created just for this event.
            parent_observation_id (Optional[str]): The ID of the parent observation, if applicable.
            name (Optional[str]): Identifier of the event. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime]): The time at which the event started, defaults to the current time.
            metadata (Optional[Any]): Additional metadata of the event. Can be any JSON object. Metadata is merged when being updated via the API.
            input (Optional[Any]): The input to the event. Can be any JSON object.
            output (Optional[Any]): The output to the event. Can be any JSON object.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): The level of the event. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the event. Additional field for context of the event. E.g. the error message of an error event.
            version (Optional[str]): The version of the event type. Used to understand how changes to the event type affect metrics. Useful in debugging.
            **kwargs: Additional keyword arguments to include in the event.

        Returns:
            StatefulSpanClient: The created event.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            trace = langfuse.trace(name = "llm-feature")

            # Create an event
            retrieval = langfuse.event(name = "retrieval", trace_id = trace.id)
            ```
        """
        event_id = id or str(uuid.uuid4())
        new_trace_id = trace_id or str(uuid.uuid4())
        self.trace_id = new_trace_id
        try:
            event_body = {
                "id": event_id,
                "trace_id": new_trace_id,
                "name": name,
                "start_time": start_time or _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "parent_observation_id": parent_observation_id,
                "version": version,
                "trace": {"release": self.release},
                **kwargs,
            }

            if trace_id is None:
                self._generate_trace(new_trace_id, name or new_trace_id)

            request = CreateEventBody(**event_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "event-create",
                "body": request.dict(exclude_none=True),
            }

            self.log.debug(f"Creating event {event}...")
            self.task_manager.add_task(event)

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulSpanClient(
                self.client,
                event_id,
                StateType.OBSERVATION,
                new_trace_id,
                self.task_manager,
            )

    def generation(
        self,
        *,
        id: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        usage: typing.Optional[typing.Union[pydantic.BaseModel, ModelUsage]] = None,
        prompt: typing.Optional[PromptClient] = None,
        **kwargs,
    ) -> "StatefulGenerationClient":
        """Create a generation.

        A generation is a span that is used to log generations of AI models. They contain additional metadata about the model, the prompt/completion, the cost of executing the model and are specifically rendered in the langfuse UI.

        Usually, you want to add a generation nested within a trace. Optionally you can nest it within another observation by providing a parent_observation_id.

        If no trace_id is provided, a new trace is created just for this generation.

        Args:
            id (Optional[str]): The id of the generation can be set, defaults to random id.
            trace_id (Optional[str]): The trace ID associated with this generation. If not provided, a new trace is created
            parent_observation_id (Optional[str]): The ID of the parent observation, if applicable.
            name (Optional[str]): Identifier of the generation. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime.datetime]): The time at which the generation started, defaults to the current time.
            end_time (Optional[datetime.datetime]): The time at which the generation ended. Automatically set by `generation.end()`.
            completion_start_time (Optional[datetime.datetime]): The time at which the completion started (streaming). Set it to get latency analytics broken down into time until completion started and completion duration.
            metadata (Optional[dict]): Additional metadata of the generation. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[str]): The level of the generation. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the generation. Additional field for context of the event. E.g. the error message of an error event.
            version (Optional[str]): The version of the generation type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            model (Optional[str]): The name of the model used for the generation.
            model_parameters (Optional[dict]): The parameters of the model used for the generation; can be any key-value pairs.
            input (Optional[dict]): The prompt used for the generation. Can be any string or JSON object.
            output (Optional[dict]): The completion generated by the model. Can be any string or JSON object.
            usage (Optional[dict]): The usage object supports the OpenAi structure with {`promptTokens`, `completionTokens`, `totalTokens`} and a more generic version {`input`, `output`, `total`, `unit`, `inputCost`, `outputCost`, `totalCost`} where unit can be of value `"TOKENS"`, `"CHARACTERS"`, `"MILLISECONDS"`, `"SECONDS"`, or `"IMAGES"`. Refer to the docs on how to [automatically infer](https://langfuse.com/docs/model-usage-and-cost) token usage and costs in Langfuse.
            prompt (Optional[PromptClient]): The Langfuse prompt object used for the generation.
            **kwargs: Additional keyword arguments to include in the generation.

        Returns:
            StatefulGenerationClient: The created generation.

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
        new_trace_id = trace_id or str(uuid.uuid4())
        new_generation_id = id or str(uuid.uuid4())
        self.trace_id = new_trace_id
        try:
            generation_body = {
                "id": new_generation_id,
                "trace_id": new_trace_id,
                "release": self.release,
                "name": name,
                "start_time": start_time or _get_timestamp(),
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
                **kwargs,
            }

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

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulGenerationClient(
                self.client,
                new_generation_id,
                StateType.OBSERVATION,
                new_trace_id,
                self.task_manager,
            )

    def _generate_trace(self, trace_id: str, name: str):
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

    def join(self):
        """Blocks until all consumer Threads are terminated. The SKD calls this upon termination of the Python Interpreter.

        If called before flushing, consumers might terminate before sending all events to Langfuse API. This method is called at exit of the SKD, right before the Python interpreter closes.
        To guarantee all messages have been delivered, you still need to call flush().
        """
        try:
            return self.task_manager.join()
        except Exception as e:
            self.log.exception(e)

    def flush(self):
        """Flush the internal event queue to the Langfuse API. It blocks until the queue is empty. It should be called when the application shuts down.

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
        """Initiate a graceful shutdown of the Langfuse SDK, ensuring all events are sent to Langfuse API and all consumer Threads are terminated.

        This function calls flush() and join() consecutively resulting in a complete shutdown of the SDK. On success of this function, no more events will be sent to Langfuse API.
        As the SDK calls join() already on shutdown, refer to flush() to ensure all events arive at the Langfuse API.
        """
        try:
            return self.task_manager.shutdown()
        except Exception as e:
            self.log.exception(e)


class StateType(Enum):
    """Enum to distinguish observation and trace states.

    Attributes:
        OBSERVATION (int): Observation state.
        TRACE (int): Trace state.
    """

    OBSERVATION = 1
    TRACE = 0


class StatefulClient(object):
    """Base class for handling stateful operations in the Langfuse system.

    This client is capable of creating different nested Langfuse objects like spans, generations, scores, and events,
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
        """Initialize the StatefulClient.

        Args:
            client (FernLangfuse): Core interface for Langfuse API interactions.
            id (str): Unique identifier of the stateful client (either observation or trace).
            state_type (StateType): Enum indicating whether the client is an observation or a trace.
            trace_id (str): Id of the trace associated with the stateful client.
            task_manager (TaskManager): Manager handling asynchronous tasks for the client.
        """
        self.client = client
        self.trace_id = trace_id
        self.id = id
        self.state_type = state_type
        self.task_manager = task_manager

    def _add_state_to_event(self, body: dict):
        if self.state_type == StateType.OBSERVATION:
            body["parent_observation_id"] = self.id
            body["trace_id"] = self.trace_id
        else:
            body["trace_id"] = self.id
        return body

    def _add_default_values(self, body: dict):
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
        level: typing.Optional[SpanLevel] = None,
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
    ) -> "StatefulGenerationClient":
        """Create a generation nested within the current observation or trace.

        A generation is a span that is used to log generations of AI models. They contain additional metadata about the model, the prompt/completion, the cost of executing the model and are specifically rendered in the langfuse UI.

        Args:
            id (Optional[str]): The id of the generation can be set, defaults to random id.
            name (Optional[str]): Identifier of the generation. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime.datetime]): The time at which the generation started, defaults to the current time.
            end_time (Optional[datetime.datetime]): The time at which the generation ended. Automatically set by `generation.end()`.
            completion_start_time (Optional[datetime.datetime]): The time at which the completion started (streaming). Set it to get latency analytics broken down into time until completion started and completion duration.
            metadata (Optional[dict]): Additional metadata of the generation. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[str]): The level of the generation. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the generation. Additional field for context of the event. E.g. the error message of an error event.
            version (Optional[str]): The version of the generation type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            model (Optional[str]): The name of the model used for the generation.
            model_parameters (Optional[dict]): The parameters of the model used for the generation; can be any key-value pairs.
            input (Optional[dict]): The prompt used for the generation. Can be any string or JSON object.
            output (Optional[dict]): The completion generated by the model. Can be any string or JSON object.
            usage (Optional[dict]): The usage object supports the OpenAi structure with {`promptTokens`, `completionTokens`, `totalTokens`} and a more generic version {`input`, `output`, `total`, `unit`, `inputCost`, `outputCost`, `totalCost`} where unit can be of value `"TOKENS"`, `"CHARACTERS"`, `"MILLISECONDS"`, `"SECONDS"`, or `"IMAGES"`. Refer to the docs on how to [automatically infer](https://langfuse.com/docs/model-usage-and-cost) token usage and costs in Langfuse.
            prompt (Optional[PromptClient]): The Langfuse prompt object used for the generation.
            **kwargs: Additional keyword arguments to include in the generation.

        Returns:
            StatefulGenerationClient: The created generation. Use this client to update the generation or create additional nested observations.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create a nested generation in Langfuse
            generation = trace.generation(
                name="summary-generation",
                model="gpt-3.5-turbo",
                model_parameters={"maxTokens": "1000", "temperature": "0.9"},
                input=[{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": "Please generate a summary of the following documents ..."}],
                metadata={"interface": "whatsapp"}
            )
            ```
        """
        generation_id = id or str(uuid.uuid4())
        try:
            generation_body = {
                "id": generation_id,
                "name": name,
                "start_time": start_time or _get_timestamp(),
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
                **kwargs,
            }

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

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulGenerationClient(
                self.client,
                generation_id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )

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
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulSpanClient":
        """Create a span nested within the current observation or trace.

        A span represents durations of units of work in a trace.

        Args:
            id (Optional[str]): The id of the span can be set, otherwise a random id is generated. Spans are upserted on id.
            name (Optional[str]): Identifier of the span. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime]): The time at which the span started, defaults to the current time.
            end_time (Optional[datetime]): The time at which the span ended. Automatically set by `span.end()`.
            metadata (Optional[dict]): Additional metadata of the span. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): The level of the span. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the span. Additional field for context of the event. E.g. the error message of an error event.
            input (Optional[dict]): The input to the span. Can be any JSON object.
            output (Optional[dict]): The output to the span. Can be any JSON object.
            version (Optional[str]): The version of the span type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            **kwargs: Additional keyword arguments to include in the span.

        Returns:
            StatefulSpanClient: The created span. Use this client to update the span or create additional nested observations.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create a span
            retrieval = langfuse.span(name = "retrieval")
            ```
        """
        span_id = id or str(uuid.uuid4())
        try:
            span_body = {
                "id": span_id,
                "name": name,
                "start_time": start_time or _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "version": version,
                "end_time": end_time,
                **kwargs,
            }

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
        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulSpanClient(
                self.client,
                span_id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )

    def score(
        self,
        *,
        id: typing.Optional[str] = None,
        name: str,
        value: float,
        comment: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulClient":
        """Create a score attached for the current observation or trace.

        Args:
            name (str): Identifier of the score.
            value (float): The value of the score. Can be any number, often standardized to 0..1
            comment (Optional[str]): Additional context/explanation of the score.
            id (Optional[str]): The id of the score. If not provided, a new UUID is generated.
            **kwargs: Additional keyword arguments to include in the score.

        Returns:
            StatefulClient: The current observation or trace for which the score was created. Passthrough for chaining.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name="example-application")

            # Add score to the trace
            trace = trace.score(
                name="user-explicit-feedback",
                value=1,
                comment="I like how personalized the response is"
            )
            ```
        """
        score_id = id or str(uuid.uuid4())
        try:
            new_score = {
                "id": score_id,
                "trace_id": self.trace_id,
                "name": name,
                "value": value,
                "comment": comment,
                **kwargs,
            }

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

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )

    def event(
        self,
        *,
        id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulClient":
        """Create an event nested within the current observation or trace.

        An event represents a discrete event in a trace.

        Args:
            id (Optional[str]): The id of the event can be set, otherwise a random id is generated.
            name (Optional[str]): Identifier of the event. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime]): The time at which the event started, defaults to the current time.
            metadata (Optional[Any]): Additional metadata of the event. Can be any JSON object. Metadata is merged when being updated via the API.
            input (Optional[Any]): The input to the event. Can be any JSON object.
            output (Optional[Any]): The output to the event. Can be any JSON object.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): The level of the event. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the event. Additional field for context of the event. E.g. the error message of an error event.
            version (Optional[str]): The version of the event type. Used to understand how changes to the event type affect metrics. Useful in debugging.
            **kwargs: Additional keyword arguments to include in the event.

        Returns:
            StatefulSpanClient: The created event. Use this client to update the event or create additional nested observations.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create an event
            retrieval = trace.event(name = "retrieval")
            ```
        """
        event_id = id or str(uuid.uuid4())
        try:
            event_body = {
                "id": event_id,
                "name": name,
                "start_time": start_time or _get_timestamp(),
                "metadata": metadata,
                "input": input,
                "output": output,
                "level": level,
                "status_message": status_message,
                "version": version,
                **kwargs,
            }

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

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulClient(
                self.client, event_id, self.state_type, self.trace_id, self.task_manager
            )

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
        """Initialize the StatefulGenerationClient."""
        super().__init__(client, id, state_type, trace_id, task_manager)

    # WHEN CHANGING THIS METHOD, UPDATE END() FUNCTION ACCORDINGLY
    def update(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        usage: typing.Optional[typing.Union[pydantic.BaseModel, ModelUsage]] = None,
        prompt: typing.Optional[PromptClient] = None,
        **kwargs,
    ) -> "StatefulGenerationClient":
        """Update the generation.

        Args:
            name (Optional[str]): Identifier of the generation. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime.datetime]): The time at which the generation started, defaults to the current time.
            end_time (Optional[datetime.datetime]): The time at which the generation ended. Automatically set by `generation.end()`.
            completion_start_time (Optional[datetime.datetime]): The time at which the completion started (streaming). Set it to get latency analytics broken down into time until completion started and completion duration.
            metadata (Optional[dict]): Additional metadata of the generation. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[str]): The level of the generation. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the generation. Additional field for context of the event. E.g. the error message of an error event.
            version (Optional[str]): The version of the generation type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            model (Optional[str]): The name of the model used for the generation.
            model_parameters (Optional[dict]): The parameters of the model used for the generation; can be any key-value pairs.
            input (Optional[dict]): The prompt used for the generation. Can be any string or JSON object.
            output (Optional[dict]): The completion generated by the model. Can be any string or JSON object.
            usage (Optional[dict]): The usage object supports the OpenAi structure with {`promptTokens`, `completionTokens`, `totalTokens`} and a more generic version {`input`, `output`, `total`, `unit`, `inputCost`, `outputCost`, `totalCost`} where unit can be of value `"TOKENS"`, `"CHARACTERS"`, `"MILLISECONDS"`, `"SECONDS"`, or `"IMAGES"`. Refer to the docs on how to [automatically infer](https://langfuse.com/docs/model-usage-and-cost) token usage and costs in Langfuse.
            prompt (Optional[PromptClient]): The Langfuse prompt object used for the generation.
            **kwargs: Additional keyword arguments to include in the generation.

        Returns:
            StatefulGenerationClient: The updated generation. Passthrough for chaining.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create a nested generation in Langfuse
            generation = trace.generation(name="summary-generation")

            # Update the generation
            generation = generation.update(metadata={"interface": "whatsapp"})
            ```
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
                **kwargs,
            }

            self.log.debug(f"Update generation {generation_body}...")

            request = UpdateGenerationBody(**generation_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "generation-update",
                "body": request.dict(exclude_none=True),
            }

            self.log.debug(f"Update generation {event}...")
            self.task_manager.add_task(event)

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulGenerationClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )

    def end(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        completion_start_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        model_parameters: typing.Optional[typing.Dict[str, MapValue]] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        usage: typing.Optional[typing.Union[pydantic.BaseModel, ModelUsage]] = None,
        prompt: typing.Optional[PromptClient] = None,
        **kwargs,
    ) -> "StatefulGenerationClient":
        """End the generation, optionally updating its properties.

        Args:
            name (Optional[str]): Identifier of the generation. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime.datetime]): The time at which the generation started, defaults to the current time.
            end_time (Optional[datetime.datetime]): Automatically set to the current time. Can be overridden to set a custom end time.
            completion_start_time (Optional[datetime.datetime]): The time at which the completion started (streaming). Set it to get latency analytics broken down into time until completion started and completion duration.
            metadata (Optional[dict]): Additional metadata of the generation. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[str]): The level of the generation. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the generation. Additional field for context of the event. E.g. the error message of an error event.
            version (Optional[str]): The version of the generation type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            model (Optional[str]): The name of the model used for the generation.
            model_parameters (Optional[dict]): The parameters of the model used for the generation; can be any key-value pairs.
            input (Optional[dict]): The prompt used for the generation. Can be any string or JSON object.
            output (Optional[dict]): The completion generated by the model. Can be any string or JSON object.
            usage (Optional[dict]): The usage object supports the OpenAi structure with {`promptTokens`, `completionTokens`, `totalTokens`} and a more generic version {`input`, `output`, `total`, `unit`, `inputCost`, `outputCost`, `totalCost`} where unit can be of value `"TOKENS"`, `"CHARACTERS"`, `"MILLISECONDS"`, `"SECONDS"`, or `"IMAGES"`. Refer to the docs on how to [automatically infer](https://langfuse.com/docs/model-usage-and-cost) token usage and costs in Langfuse.
            prompt (Optional[PromptClient]): The Langfuse prompt object used for the generation.
            **kwargs: Additional keyword arguments to include in the generation.

        Returns:
            StatefulGenerationClient: The ended generation. Passthrough for chaining.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create a nested generation in Langfuse
            generation = trace.generation(name="summary-generation")

            # End the generation and update its properties
            generation = generation.end(metadata={"interface": "whatsapp"})
            ```
        """
        return self.update(
            name=name,
            start_time=start_time,
            end_time=end_time or _get_timestamp(),
            metadata=metadata,
            level=level,
            status_message=status_message,
            version=version,
            completion_start_time=completion_start_time,
            model=model,
            model_parameters=model_parameters,
            input=input,
            output=output,
            usage=usage,
            prompt=prompt,
            **kwargs,
        )


class StatefulSpanClient(StatefulClient):
    """Class for handling stateful operations of spans in the Langfuse system. Inherits from StatefulClient.

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
        """Initialize the StatefulSpanClient."""
        super().__init__(client, id, state_type, trace_id, task_manager)

    # WHEN CHANGING THIS METHOD, UPDATE END() FUNCTION ACCORDINGLY
    def update(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulSpanClient":
        """Update the span.

        Args:
            name (Optional[str]): Identifier of the span. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime]): The time at which the span started, defaults to the current time.
            end_time (Optional[datetime]): The time at which the span ended. Automatically set by `span.end()`.
            metadata (Optional[dict]): Additional metadata of the span. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): The level of the span. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the span. Additional field for context of the event. E.g. the error message of an error event.
            input (Optional[dict]): The input to the span. Can be any JSON object.
            output (Optional[dict]): The output to the span. Can be any JSON object.
            version (Optional[str]): The version of the span type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            **kwargs: Additional keyword arguments to include in the span.

        Returns:
            StatefulSpanClient: The updated span. Passthrough for chaining.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create a nested span in Langfuse
            span = trace.span(name="retrieval")

            # Update the span
            span = span.update(metadata={"interface": "whatsapp"})
            ```
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
                **kwargs,
            }
            self.log.debug(f"Update span {span_body}...")

            request = UpdateSpanBody(**span_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "span-update",
                "body": request.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)
        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulSpanClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )

    def end(
        self,
        *,
        name: typing.Optional[str] = None,
        start_time: typing.Optional[dt.datetime] = None,
        end_time: typing.Optional[dt.datetime] = None,
        metadata: typing.Optional[typing.Any] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        level: typing.Optional[SpanLevel] = None,
        status_message: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        **kwargs,
    ) -> "StatefulSpanClient":
        """End the span, optionally updating its properties.

        Args:
            name (Optional[str]): Identifier of the span. Useful for sorting/filtering in the UI.
            start_time (Optional[datetime]): The time at which the span started, defaults to the current time.
            end_time (Optional[datetime]): The time at which the span ended. Automatically set by `span.end()`.
            metadata (Optional[dict]): Additional metadata of the span. Can be any JSON object. Metadata is merged when being updated via the API.
            level (Optional[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]]): The level of the span. Can be `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
            status_message (Optional[str]): The status message of the span. Additional field for context of the event. E.g. the error message of an error event.
            input (Optional[dict]): The input to the span. Can be any JSON object.
            output (Optional[dict]): The output to the span. Can be any JSON object.
            version (Optional[str]): The version of the span type. Used to understand how changes to the span type affect metrics. Useful in debugging.
            **kwargs: Additional keyword arguments to include in the span.

        Returns:
            StatefulSpanClient: The updated span. Passthrough for chaining.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Create a nested span in Langfuse
            span = trace.span(name="retrieval")

            # End the span and update its properties
            span = span.end(metadata={"interface": "whatsapp"})
            ```
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
                "end_time": end_time or _get_timestamp(),
                **kwargs,
            }
            return self.update(**span_body)

        except Exception as e:
            self.log.warning(e)
        finally:
            return StatefulSpanClient(
                self.client,
                self.id,
                StateType.OBSERVATION,
                self.trace_id,
                task_manager=self.task_manager,
            )

    def get_langchain_handler(self):
        """Get langchain callback handler associated with the current span.

        Returns:
            CallbackHandler: An instance of CallbackHandler linked to this StatefulSpanClient.
        """
        from langfuse.callback import CallbackHandler

        return CallbackHandler(stateful_client=self)


class StatefulTraceClient(StatefulClient):
    """Class for handling stateful operations of traces in the Langfuse system. Inherits from StatefulClient.

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
        """Initialize the StatefulTraceClient."""
        super().__init__(client, id, state_type, trace_id, task_manager)
        self.task_manager = task_manager

    def update(
        self,
        *,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        release: typing.Optional[str] = None,
        input: typing.Optional[typing.Any] = None,
        output: typing.Optional[typing.Any] = None,
        metadata: typing.Optional[typing.Any] = None,
        tags: typing.Optional[typing.List[str]] = None,
        public: typing.Optional[bool] = None,
        **kwargs,
    ) -> "StatefulTraceClient":
        """Update the trace.

        Args:
            name: Identifier of the trace. Useful for sorting/filtering in the UI.
            input: The input of the trace. Can be any JSON object.
            output: The output of the trace. Can be any JSON object.
            metadata: Additional metadata of the trace. Can be any JSON object. Metadata is merged when being updated via the API.
            user_id: The id of the user that triggered the execution. Used to provide user-level analytics.
            session_id: Used to group multiple traces into a session in Langfuse. Use your own session/thread identifier.
            version: The version of the trace type. Used to understand how changes to the trace type affect metrics. Useful in debugging.
            release: The release identifier of the current deployment. Used to understand how changes of different deployments affect metrics. Useful in debugging.
            tags: Tags are used to categorize or label traces. Traces can be filtered by tags in the UI and GET API. Tags can also be changed in the UI. Tags are merged and never deleted via the API.
            public: You can make a trace `public` to share it via a public link. This allows others to view the trace without needing to log in or be members of your Langfuse project.
            **kwargs: Additional keyword arguments that can be included in the trace.

        Returns:
            StatefulTraceClient: The updated trace. Passthrough for chaining.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(
                name="example-application",
                user_id="user-1234")
            )

            # Update the trace
            trace = trace.update(
                output={"result": "success"},
                metadata={"interface": "whatsapp"}
            )
            ```
        """
        try:
            trace_body = {
                "id": self.id,
                "name": name,
                "userId": user_id,
                "sessionId": session_id
                or kwargs.get("sessionId", None),  # backward compatibility
                "version": version,
                "release": release,
                "input": input,
                "output": output,
                "metadata": metadata,
                "public": public,
                "tags": tags,
                **kwargs,
            }
            self.log.debug(f"Update trace {trace_body}...")

            request = TraceBody(**trace_body)

            event = {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "body": request.dict(exclude_none=True),
            }

            self.task_manager.add_task(event)

        except Exception as e:
            self.log.exception(e)
        finally:
            return StatefulTraceClient(
                self.client,
                self.id,
                StateType.TRACE,
                self.trace_id,
                task_manager=self.task_manager,
            )

    def get_langchain_handler(self):
        """Get langchain callback handler associated with the current trace.

        This method creates and returns a CallbackHandler instance, linking it with the current
        trace. Use this if you want to group multiple Langchain runs within a single trace.

        Raises:
            ImportError: If the 'langchain' module is not installed, indicating missing functionality.

        Returns:
            CallbackHandler: Langchain callback handler linked to the current trace.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Create a trace
            trace = langfuse.trace(name = "llm-feature")

            # Get a langchain callback handler
            handler = trace.get_langchain_handler()
            ```
        """
        try:
            from langfuse.callback import CallbackHandler

            self.log.debug(f"Creating new handler for trace {self.id}")

            return CallbackHandler(
                stateful_client=self, debug=self.log.level == logging.DEBUG
            )
        except Exception as e:
            self.log.exception(e)

    def getNewHandler(self):
        """Alias for the `get_langchain_handler` method. Retrieves a callback handler for the trace. Deprecated."""
        return self.get_langchain_handler()


class DatasetItemClient:
    """Class for managing dataset items in Langfuse.

    Args:
        id (str): Unique identifier of the dataset item.
        status (DatasetStatus): The status of the dataset item. Can be either 'ACTIVE' or 'ARCHIVED'.
        input (Any): Input data of the dataset item.
        expected_output (Optional[Any]): Expected output of the dataset item.
        source_observation_id (Optional[str]): Identifier of the source observation.
        dataset_id (str): Identifier of the dataset to which this item belongs.
        created_at (datetime): Timestamp of dataset item creation.
        updated_at (datetime): Timestamp of the last update to the dataset item.
        langfuse (Langfuse): Instance of Langfuse client for API interactions.

    Example:
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
        """Initialize the DatasetItemClient."""
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
        """Link the dataset item to observation within a specific dataset run. Creates a dataset run item.

        Args:
            observation (Union[StatefulClient, str]): The observation to link, either as a client or as an ID.
            run_name (str): The name of the dataset run.
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

    Example:
        Print the input of each dataset item in a dataset.
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
        """Initialize the DatasetClient."""
        self.id = dataset.id
        self.name = dataset.name
        self.project_id = dataset.project_id
        self.dataset_name = dataset.name
        self.created_at = dataset.created_at
        self.updated_at = dataset.updated_at
        self.items = items
        self.runs = dataset.runs
