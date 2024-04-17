# This file was auto-generated by Fern from our API Definition.

import typing
import urllib.parse
from json.decoder import JSONDecodeError

from ...core.api_error import ApiError
from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ...core.jsonable_encoder import jsonable_encoder
from ...core.pydantic_utilities import pydantic_v1
from ...core.remove_none_from_dict import remove_none_from_dict
from ...core.request_options import RequestOptions
from ..commons.errors.access_denied_error import AccessDeniedError
from ..commons.errors.error import Error
from ..commons.errors.method_not_allowed_error import MethodNotAllowedError
from ..commons.errors.not_found_error import NotFoundError
from ..commons.errors.unauthorized_error import UnauthorizedError
from ..commons.types.session_with_traces import SessionWithTraces


class SessionsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def get(
        self,
        session_id: str,
        *,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> SessionWithTraces:
        """
        Get a session

        Parameters:
            - session_id: str. The unique id of a session

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
        client.sessions.get(
            session_id="string",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/",
                f"api/public/sessions/{jsonable_encoder(session_id)}",
            ),
            params=jsonable_encoder(
                request_options.get("additional_query_parameters")
                if request_options is not None
                else None
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(
                            request_options.get("additional_headers", {})
                            if request_options is not None
                            else {}
                        ),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None
            and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries")
            if request_options is not None
            else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(SessionWithTraces, _response.json())  # type: ignore
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
            raise NotFoundError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncSessionsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def get(
        self,
        session_id: str,
        *,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> SessionWithTraces:
        """
        Get a session

        Parameters:
            - session_id: str. The unique id of a session

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
        await client.sessions.get(
            session_id="string",
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/",
                f"api/public/sessions/{jsonable_encoder(session_id)}",
            ),
            params=jsonable_encoder(
                request_options.get("additional_query_parameters")
                if request_options is not None
                else None
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(
                            request_options.get("additional_headers", {})
                            if request_options is not None
                            else {}
                        ),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None
            and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries")
            if request_options is not None
            else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(SessionWithTraces, _response.json())  # type: ignore
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
            raise NotFoundError(pydantic_v1.parse_obj_as(typing.Any, _response.json()))  # type: ignore
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
