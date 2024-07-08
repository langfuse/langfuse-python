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
from ..commons.types.score import Score
from ..commons.types.score_data_type import ScoreDataType
from ..commons.types.score_source import ScoreSource
from .types.create_score_request import CreateScoreRequest
from .types.scores import Scores

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class ScoreClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def create(
        self,
        *,
        request: CreateScoreRequest,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Score:
        """
        Create a score

        Parameters:
            - request: CreateScoreRequest.

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        from finto import CreateScoreRequest
        from finto.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.score.create(
            request=CreateScoreRequest(
                name="novelty",
                value=0.9,
                trace_id="cdef-1234-5678-90ab",
            ),
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "POST",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", "api/public/scores"
            ),
            params=jsonable_encoder(
                request_options.get("additional_query_parameters")
                if request_options is not None
                else None
            ),
            json=jsonable_encoder(request)
            if request_options is None
            or request_options.get("additional_body_parameters") is None
            else {
                **jsonable_encoder(request),
                **(
                    jsonable_encoder(
                        remove_none_from_dict(
                            request_options.get("additional_body_parameters", {})
                        )
                    )
                ),
            },
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
            return pydantic_v1.parse_obj_as(Score, _response.json())  # type: ignore
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

    def get(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        user_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        from_timestamp: typing.Optional[dt.datetime] = None,
        source: typing.Optional[ScoreSource] = None,
        operator: typing.Optional[str] = None,
        value: typing.Optional[float] = None,
        score_ids: typing.Optional[str] = None,
        config_id: typing.Optional[str] = None,
        data_type: typing.Optional[ScoreDataType] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Scores:
        """
        Get a list of scores

        Parameters:
            - page: typing.Optional[int]. Page number, starts at 1.

            - limit: typing.Optional[int]. Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.

            - user_id: typing.Optional[str]. Retrieve only scores with this userId associated to the trace.

            - name: typing.Optional[str]. Retrieve only scores with this name.

            - from_timestamp: typing.Optional[dt.datetime]. Retrieve only scores newer than this datetime (ISO 8601).

            - source: typing.Optional[ScoreSource]. Retrieve only scores from a specific source.

            - operator: typing.Optional[str]. Retrieve only scores with <operator> value.

            - value: typing.Optional[float]. Retrieve only scores with <operator> value.

            - score_ids: typing.Optional[str]. Comma-separated list of score IDs to limit the results to.

            - config_id: typing.Optional[str]. Retrieve only scores with a specific configId.

            - data_type: typing.Optional[ScoreDataType]. Retrieve only scores with a specific dataType.

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        import datetime

        from finto import ScoreDataType, ScoreSource
        from finto.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.score.get(
            page=1,
            limit=1,
            user_id="string",
            name="string",
            from_timestamp=datetime.datetime.fromisoformat(
                "2024-01-15 09:30:00+00:00",
            ),
            source=ScoreSource.ANNOTATION,
            operator="string",
            value=1.1,
            score_ids="string",
            config_id="string",
            data_type=ScoreDataType.NUMERIC,
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", "api/public/scores"
            ),
            params=jsonable_encoder(
                remove_none_from_dict(
                    {
                        "page": page,
                        "limit": limit,
                        "userId": user_id,
                        "name": name,
                        "fromTimestamp": serialize_datetime(from_timestamp)
                        if from_timestamp is not None
                        else None,
                        "source": source,
                        "operator": operator,
                        "value": value,
                        "scoreIds": score_ids,
                        "configId": config_id,
                        "dataType": data_type,
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
            return pydantic_v1.parse_obj_as(Scores, _response.json())  # type: ignore
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

    def get_by_id(
        self, score_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> Score:
        """
        Get a score

        Parameters:
            - score_id: str. The unique langfuse identifier of a score

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
        client.score.get_by_id(
            score_id="string",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/",
                f"api/public/scores/{jsonable_encoder(score_id)}",
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
            return pydantic_v1.parse_obj_as(Score, _response.json())  # type: ignore
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

    def delete(
        self, score_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> None:
        """
        Delete a score

        Parameters:
            - score_id: str. The unique langfuse identifier of a score

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
        client.score.delete(
            score_id="string",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "DELETE",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/",
                f"api/public/scores/{jsonable_encoder(score_id)}",
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
            return
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


class AsyncScoreClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def create(
        self,
        *,
        request: CreateScoreRequest,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Score:
        """
        Create a score

        Parameters:
            - request: CreateScoreRequest.

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        from finto import CreateScoreRequest
        from finto.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        await client.score.create(
            request=CreateScoreRequest(
                name="novelty",
                value=0.9,
                trace_id="cdef-1234-5678-90ab",
            ),
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "POST",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", "api/public/scores"
            ),
            params=jsonable_encoder(
                request_options.get("additional_query_parameters")
                if request_options is not None
                else None
            ),
            json=jsonable_encoder(request)
            if request_options is None
            or request_options.get("additional_body_parameters") is None
            else {
                **jsonable_encoder(request),
                **(
                    jsonable_encoder(
                        remove_none_from_dict(
                            request_options.get("additional_body_parameters", {})
                        )
                    )
                ),
            },
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
            return pydantic_v1.parse_obj_as(Score, _response.json())  # type: ignore
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

    async def get(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        user_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        from_timestamp: typing.Optional[dt.datetime] = None,
        source: typing.Optional[ScoreSource] = None,
        operator: typing.Optional[str] = None,
        value: typing.Optional[float] = None,
        score_ids: typing.Optional[str] = None,
        config_id: typing.Optional[str] = None,
        data_type: typing.Optional[ScoreDataType] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Scores:
        """
        Get a list of scores

        Parameters:
            - page: typing.Optional[int]. Page number, starts at 1.

            - limit: typing.Optional[int]. Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.

            - user_id: typing.Optional[str]. Retrieve only scores with this userId associated to the trace.

            - name: typing.Optional[str]. Retrieve only scores with this name.

            - from_timestamp: typing.Optional[dt.datetime]. Retrieve only scores newer than this datetime (ISO 8601).

            - source: typing.Optional[ScoreSource]. Retrieve only scores from a specific source.

            - operator: typing.Optional[str]. Retrieve only scores with <operator> value.

            - value: typing.Optional[float]. Retrieve only scores with <operator> value.

            - score_ids: typing.Optional[str]. Comma-separated list of score IDs to limit the results to.

            - config_id: typing.Optional[str]. Retrieve only scores with a specific configId.

            - data_type: typing.Optional[ScoreDataType]. Retrieve only scores with a specific dataType.

            - request_options: typing.Optional[RequestOptions]. Request-specific configuration.
        ---
        import datetime

        from finto import ScoreDataType, ScoreSource
        from finto.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        await client.score.get(
            page=1,
            limit=1,
            user_id="string",
            name="string",
            from_timestamp=datetime.datetime.fromisoformat(
                "2024-01-15 09:30:00+00:00",
            ),
            source=ScoreSource.ANNOTATION,
            operator="string",
            value=1.1,
            score_ids="string",
            config_id="string",
            data_type=ScoreDataType.NUMERIC,
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", "api/public/scores"
            ),
            params=jsonable_encoder(
                remove_none_from_dict(
                    {
                        "page": page,
                        "limit": limit,
                        "userId": user_id,
                        "name": name,
                        "fromTimestamp": serialize_datetime(from_timestamp)
                        if from_timestamp is not None
                        else None,
                        "source": source,
                        "operator": operator,
                        "value": value,
                        "scoreIds": score_ids,
                        "configId": config_id,
                        "dataType": data_type,
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
            return pydantic_v1.parse_obj_as(Scores, _response.json())  # type: ignore
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

    async def get_by_id(
        self, score_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> Score:
        """
        Get a score

        Parameters:
            - score_id: str. The unique langfuse identifier of a score

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
        await client.score.get_by_id(
            score_id="string",
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "GET",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/",
                f"api/public/scores/{jsonable_encoder(score_id)}",
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
            return pydantic_v1.parse_obj_as(Score, _response.json())  # type: ignore
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

    async def delete(
        self, score_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> None:
        """
        Delete a score

        Parameters:
            - score_id: str. The unique langfuse identifier of a score

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
        await client.score.delete(
            score_id="string",
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            "DELETE",
            urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/",
                f"api/public/scores/{jsonable_encoder(score_id)}",
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
            return
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
