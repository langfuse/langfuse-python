from typing import Optional, Union
import logging
import os

from langfuse.client import Langfuse, StatefulTraceClient, StatefulSpanClient, StateType


class LangfuseBaseCallbackHandler:
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
        stateful_client: Optional[
            Union[StatefulTraceClient, StatefulSpanClient]
        ] = None,
        version: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        release: Optional[str] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        sdk_integration: str,
    ) -> None:
        self.version = version
        self.session_id = session_id
        self.user_id = user_id
        self.trace_name = trace_name

        self.root_span = None
        self.langfuse = None

        prio_public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        prio_secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        prio_host = host or os.environ.get(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )

        if stateful_client and isinstance(stateful_client, StatefulTraceClient):
            self.trace = stateful_client
            self._task_manager = stateful_client.task_manager

        elif stateful_client and isinstance(stateful_client, StatefulSpanClient):
            self.root_span = stateful_client
            self.trace = StatefulTraceClient(
                stateful_client.client,
                stateful_client.trace_id,
                StateType.TRACE,
                stateful_client.trace_id,
                stateful_client.task_manager,
            )
            self._task_manager = stateful_client.task_manager

        elif prio_public_key and prio_secret_key:
            args = {
                "public_key": prio_public_key,
                "secret_key": prio_secret_key,
                "host": prio_host,
                "debug": debug,
            }

            if release is not None:
                args["release"] = release
            if threads is not None:
                args["threads"] = threads
            if flush_at is not None:
                args["flush_at"] = flush_at
            if flush_interval is not None:
                args["flush_interval"] = flush_interval
            if max_retries is not None:
                args["max_retries"] = max_retries
            if timeout is not None:
                args["timeout"] = timeout

            args["sdk_integration"] = sdk_integration

            self.langfuse = Langfuse(**args)
            self.trace: Optional[StatefulTraceClient] = None
            self._task_manager = self.langfuse.task_manager

        else:
            self.log.error(
                "Either provide a stateful langfuse object or both public_key and secret_key."
            )
            raise ValueError(
                "Either provide a stateful langfuse object or both public_key and secret_key."
            )

    def get_trace_id(self):
        return self.trace.id if self.trace else None

    def get_trace_url(self):
        return self.trace.get_trace_url() if self.trace else None

    def flush(self):
        if self.trace is not None:
            self.trace.task_manager.flush()
        elif self.root_span is not None:
            self.root_span.task_manager.flush()
        else:
            self.log.debug("There was no trace yet, hence no flushing possible.")

    def auth_check(self):
        if self.langfuse is not None:
            return self.langfuse.auth_check()
        elif self.trace is not None:
            projects = self.trace.client.projects.get()
            if len(projects.data) == 0:
                raise Exception("No projects found for the keys.")
            return True
        elif self.root_span is not None:
            projects = self.root_span.client.projects.get()
            if len(projects) == 0:
                raise Exception("No projects found for the keys.")
            return True

        return False
