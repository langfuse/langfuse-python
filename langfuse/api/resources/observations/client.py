# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing
import urllib.parse
from json.decoder import JSONDecodeError

from ...core.api_error import ApiError
from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ...core.datetime_utils import serialize_datetime
from ...core.jsonable_encoder import jsonable_encoder
from ...core.pydantic_utilities import pydantic_v1
from ...core.remove_none_from_dict import remove_none_from_dict
from ...core.request_options import RequestOptions
from ..commons.errors.access_denied_error import AccessDeniedError
from ..commons.errors.error import Error
from ..commons.errors.method_not_allowed_error import MethodNotAllowedError
from ..commons.errors.not_found_error import NotFoundError
from ..commons.errors.unauthorized_error import UnauthorizedError
from ..commons.types.observations_view import ObservationsView
from .types.observations_views import ObservationsViews


class ObservationsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def get(self, observation_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> ObservationsView:
        """Get a observation

        Parameters:
            - observation_id: str. The unique langfuse identifier of an observation, can be an event, span or generation

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        from finto.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.observations.get(
            observation_id="string",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", f"api/public/observations/{jsonable_encoder(observation_id)}"
            ),
            params=jsonable_encoder(
                request_options.get("additional_query_parameters") if request_options is not None else None
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(ObservationsView, _response.json())  # type: ignore
        if _response.status_code == 400:
            raise Error(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 401:
            raise UnauthorizedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 403:
            raise AccessDeniedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 405:
            raise MethodNotAllowedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 404:
            raise NotFoundError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def get_many(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        type: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        from_start_time: typing.Optional[dt.datetime] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> ObservationsViews:
        """Get a list of observations

        Parameters:
            - page: typing.Optional[int].

            - limit: typing.Optional[int].

            - name: typing.Optional[str].

            - user_id: typing.Optional[str].

            - type: typing.Optional[str].

            - trace_id: typing.Optional[str].

            - parent_observation_id: typing.Optional[str].

            - from_start_time: typing.Optional[dt.datetime]. Retrieve only observations with a start_time greater than this timestamp.

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        import datetime

        from finto.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.observations.get_many(
            page=1,
            limit=1,
            name="string",
            user_id="string",
            type="string",
            trace_id="string",
            parent_observation_id="string",
            from_start_time=datetime.datetime.fromisoformat(
                "2024-01-15 09:30:00+00:00",
            ),
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", "api/public/observations"),
            params=jsonable_encoder(
                remove_none_from_dict(
                    {
                        "page": page,
                        "limit": limit,
                        "name": name,
                        "userId": user_id,
                        "type": type,
                        "traceId": trace_id,
                        "parentObservationId": parent_observation_id,
                        "fromStartTime": serialize_datetime(from_start_time) if from_start_time is not None else None,
                        **(
                            request_options.get("additional_query_parameters", {})
                            if request_options is not None
                            else {}
                        ),
                    }
                )
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(ObservationsViews, _response.json())  # type: ignore
        if _response.status_code == 400:
            raise Error(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 401:
            raise UnauthorizedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 403:
            raise AccessDeniedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 405:
            raise MethodNotAllowedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 404:
            raise NotFoundError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncObservationsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def get(
        self, observation_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> ObservationsView:
        """Get a observation

        Parameters:
            - observation_id: str. The unique langfuse identifier of an observation, can be an event, span or generation

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        from finto.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        await client.observations.get(
            observation_id="string",
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", f"api/public/observations/{jsonable_encoder(observation_id)}"
            ),
            params=jsonable_encoder(
                request_options.get("additional_query_parameters") if request_options is not None else None
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(ObservationsView, _response.json())  # type: ignore
        if _response.status_code == 400:
            raise Error(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 401:
            raise UnauthorizedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 403:
            raise AccessDeniedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 405:
            raise MethodNotAllowedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 404:
            raise NotFoundError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def get_many(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        type: typing.Optional[str] = None,
        trace_id: typing.Optional[str] = None,
        parent_observation_id: typing.Optional[str] = None,
        from_start_time: typing.Optional[dt.datetime] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> ObservationsViews:
        """Get a list of observations

        Parameters:
            - page: typing.Optional[int].

            - limit: typing.Optional[int].

            - name: typing.Optional[str].

            - user_id: typing.Optional[str].

            - type: typing.Optional[str].

            - trace_id: typing.Optional[str].

            - parent_observation_id: typing.Optional[str].

            - from_start_time: typing.Optional[dt.datetime]. Retrieve only observations with a start_time greater than this timestamp.

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        import datetime

        from finto.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        await client.observations.get_many(
            page=1,
            limit=1,
            name="string",
            user_id="string",
            type="string",
            trace_id="string",
            parent_observation_id="string",
            from_start_time=datetime.datetime.fromisoformat(
                "2024-01-15 09:30:00+00:00",
            ),
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", "api/public/observations"),
            params=jsonable_encoder(
                remove_none_from_dict(
                    {
                        "page": page,
                        "limit": limit,
                        "name": name,
                        "userId": user_id,
                        "type": type,
                        "traceId": trace_id,
                        "parentObservationId": parent_observation_id,
                        "fromStartTime": serialize_datetime(from_start_time) if from_start_time is not None else None,
                        **(
                            request_options.get("additional_query_parameters", {})
                            if request_options is not None
                            else {}
                        ),
                    }
                )
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(ObservationsViews, _response.json())  # type: ignore
        if _response.status_code == 400:
            raise Error(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 401:
            raise UnauthorizedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 403:
            raise AccessDeniedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 405:
            raise MethodNotAllowedError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        if _response.status_code == 404:
            raise NotFoundError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
