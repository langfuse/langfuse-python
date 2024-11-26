# This file was auto-generated by Fern from our API Definition.

import typing
from json.decoder import JSONDecodeError

from ...core.api_error import ApiError
from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ...core.pydantic_utilities import pydantic_v1
from ...core.request_options import RequestOptions
from ..commons.errors.access_denied_error import AccessDeniedError
from ..commons.errors.error import Error
from ..commons.errors.method_not_allowed_error import MethodNotAllowedError
from ..commons.errors.not_found_error import NotFoundError
from ..commons.errors.unauthorized_error import UnauthorizedError
from .types.ingestion_event import IngestionEvent
from .types.ingestion_response import IngestionResponse

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class IngestionClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def batch(
        self,
        *,
        batch: typing.Sequence[IngestionEvent],
        metadata: typing.Optional[typing.Any] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> IngestionResponse:
        """
        Batched ingestion for Langfuse Tracing. If you want to use tracing via the API, such as to build your own Langfuse client implementation, this is the only API route you need to implement.

        Notes:

        - Batch sizes are limited to 3.5 MB in total. You need to adjust the number of events per batch accordingly.
        - The API does not return a 4xx status code for input errors. Instead, it responds with a 207 status code, which includes a list of the encountered errors.

        Parameters
        ----------
        batch : typing.Sequence[IngestionEvent]
            Batch of tracing events to be ingested. Discriminated by attribute `type`.

        metadata : typing.Optional[typing.Any]
            Optional. Metadata field used by the Langfuse SDKs for debugging.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        IngestionResponse

        Examples
        --------
        import datetime

        from finto import IngestionEvent_TraceCreate, TraceBody
        from langfuse.api.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.ingestion.batch(
            batch=[
                IngestionEvent_TraceCreate(
                    body=TraceBody(
                        id="string",
                        timestamp=datetime.datetime.fromisoformat(
                            "2024-01-15 09:30:00+00:00",
                        ),
                        name="string",
                        user_id="string",
                        input={"key": "value"},
                        output={"key": "value"},
                        session_id="string",
                        release="string",
                        version="string",
                        metadata={"key": "value"},
                        tags=["string"],
                        public=True,
                    ),
                    id="string",
                    timestamp="string",
                    metadata={"key": "value"},
                )
            ],
            metadata={"key": "value"},
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "api/public/ingestion",
            method="POST",
            json={"batch": batch, "metadata": metadata},
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(IngestionResponse, _response.json())  # type: ignore
            if _response.status_code == 400:
                raise Error(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
            if _response.status_code == 401:
                raise UnauthorizedError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            if _response.status_code == 403:
                raise AccessDeniedError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            if _response.status_code == 404:
                raise NotFoundError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncIngestionClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def batch(
        self,
        *,
        batch: typing.Sequence[IngestionEvent],
        metadata: typing.Optional[typing.Any] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> IngestionResponse:
        """
        Batched ingestion for Langfuse Tracing. If you want to use tracing via the API, such as to build your own Langfuse client implementation, this is the only API route you need to implement.

        Notes:

        - Batch sizes are limited to 3.5 MB in total. You need to adjust the number of events per batch accordingly.
        - The API does not return a 4xx status code for input errors. Instead, it responds with a 207 status code, which includes a list of the encountered errors.

        Parameters
        ----------
        batch : typing.Sequence[IngestionEvent]
            Batch of tracing events to be ingested. Discriminated by attribute `type`.

        metadata : typing.Optional[typing.Any]
            Optional. Metadata field used by the Langfuse SDKs for debugging.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        IngestionResponse

        Examples
        --------
        import asyncio
        import datetime

        from finto import IngestionEvent_TraceCreate, TraceBody
        from langfuse.api.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.ingestion.batch(
                batch=[
                    IngestionEvent_TraceCreate(
                        body=TraceBody(
                            id="string",
                            timestamp=datetime.datetime.fromisoformat(
                                "2024-01-15 09:30:00+00:00",
                            ),
                            name="string",
                            user_id="string",
                            input={"key": "value"},
                            output={"key": "value"},
                            session_id="string",
                            release="string",
                            version="string",
                            metadata={"key": "value"},
                            tags=["string"],
                            public=True,
                        ),
                        id="string",
                        timestamp="string",
                        metadata={"key": "value"},
                    )
                ],
                metadata={"key": "value"},
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "api/public/ingestion",
            method="POST",
            json={"batch": batch, "metadata": metadata},
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(IngestionResponse, _response.json())  # type: ignore
            if _response.status_code == 400:
                raise Error(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
            if _response.status_code == 401:
                raise UnauthorizedError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            if _response.status_code == 403:
                raise AccessDeniedError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            if _response.status_code == 404:
                raise NotFoundError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
