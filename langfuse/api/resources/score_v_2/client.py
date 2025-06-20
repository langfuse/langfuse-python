# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing
from json.decoder import JSONDecodeError

from ...core.api_error import ApiError
from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ...core.datetime_utils import serialize_datetime
from ...core.jsonable_encoder import jsonable_encoder
from ...core.pydantic_utilities import pydantic_v1
from ...core.request_options import RequestOptions
from ..commons.errors.access_denied_error import AccessDeniedError
from ..commons.errors.error import Error
from ..commons.errors.method_not_allowed_error import MethodNotAllowedError
from ..commons.errors.not_found_error import NotFoundError
from ..commons.errors.unauthorized_error import UnauthorizedError
from ..commons.types.score import Score
from ..commons.types.score_data_type import ScoreDataType
from ..commons.types.score_source import ScoreSource
from .types.get_scores_response import GetScoresResponse


class ScoreV2Client:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def get(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        user_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        from_timestamp: typing.Optional[dt.datetime] = None,
        to_timestamp: typing.Optional[dt.datetime] = None,
        environment: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        source: typing.Optional[ScoreSource] = None,
        operator: typing.Optional[str] = None,
        value: typing.Optional[float] = None,
        score_ids: typing.Optional[str] = None,
        config_id: typing.Optional[str] = None,
        queue_id: typing.Optional[str] = None,
        data_type: typing.Optional[ScoreDataType] = None,
        trace_tags: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> GetScoresResponse:
        """
        Get a list of scores (supports both trace and session scores)

        Parameters
        ----------
        page : typing.Optional[int]
            Page number, starts at 1.

        limit : typing.Optional[int]
            Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.

        user_id : typing.Optional[str]
            Retrieve only scores with this userId associated to the trace.

        name : typing.Optional[str]
            Retrieve only scores with this name.

        from_timestamp : typing.Optional[dt.datetime]
            Optional filter to only include scores created on or after a certain datetime (ISO 8601)

        to_timestamp : typing.Optional[dt.datetime]
            Optional filter to only include scores created before a certain datetime (ISO 8601)

        environment : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            Optional filter for scores where the environment is one of the provided values.

        source : typing.Optional[ScoreSource]
            Retrieve only scores from a specific source.

        operator : typing.Optional[str]
            Retrieve only scores with <operator> value.

        value : typing.Optional[float]
            Retrieve only scores with <operator> value.

        score_ids : typing.Optional[str]
            Comma-separated list of score IDs to limit the results to.

        config_id : typing.Optional[str]
            Retrieve only scores with a specific configId.

        queue_id : typing.Optional[str]
            Retrieve only scores with a specific annotation queueId.

        data_type : typing.Optional[ScoreDataType]
            Retrieve only scores with a specific dataType.

        trace_tags : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            Only scores linked to traces that include all of these tags will be returned.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        GetScoresResponse

        Examples
        --------
        from langfuse.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.score_v_2.get()
        """
        _response = self._client_wrapper.httpx_client.request(
            "api/public/v2/scores",
            method="GET",
            params={
                "page": page,
                "limit": limit,
                "userId": user_id,
                "name": name,
                "fromTimestamp": serialize_datetime(from_timestamp)
                if from_timestamp is not None
                else None,
                "toTimestamp": serialize_datetime(to_timestamp)
                if to_timestamp is not None
                else None,
                "environment": environment,
                "source": source,
                "operator": operator,
                "value": value,
                "scoreIds": score_ids,
                "configId": config_id,
                "queueId": queue_id,
                "dataType": data_type,
                "traceTags": trace_tags,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(GetScoresResponse, _response.json())  # type: ignore
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

    def get_by_id(
        self, score_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> Score:
        """
        Get a score (supports both trace and session scores)

        Parameters
        ----------
        score_id : str
            The unique langfuse identifier of a score

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Score

        Examples
        --------
        from langfuse.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.score_v_2.get_by_id(
            score_id="scoreId",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            f"api/public/v2/scores/{jsonable_encoder(score_id)}",
            method="GET",
            request_options=request_options,
        )
        try:
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
                raise NotFoundError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncScoreV2Client:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def get(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        user_id: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        from_timestamp: typing.Optional[dt.datetime] = None,
        to_timestamp: typing.Optional[dt.datetime] = None,
        environment: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        source: typing.Optional[ScoreSource] = None,
        operator: typing.Optional[str] = None,
        value: typing.Optional[float] = None,
        score_ids: typing.Optional[str] = None,
        config_id: typing.Optional[str] = None,
        queue_id: typing.Optional[str] = None,
        data_type: typing.Optional[ScoreDataType] = None,
        trace_tags: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> GetScoresResponse:
        """
        Get a list of scores (supports both trace and session scores)

        Parameters
        ----------
        page : typing.Optional[int]
            Page number, starts at 1.

        limit : typing.Optional[int]
            Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.

        user_id : typing.Optional[str]
            Retrieve only scores with this userId associated to the trace.

        name : typing.Optional[str]
            Retrieve only scores with this name.

        from_timestamp : typing.Optional[dt.datetime]
            Optional filter to only include scores created on or after a certain datetime (ISO 8601)

        to_timestamp : typing.Optional[dt.datetime]
            Optional filter to only include scores created before a certain datetime (ISO 8601)

        environment : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            Optional filter for scores where the environment is one of the provided values.

        source : typing.Optional[ScoreSource]
            Retrieve only scores from a specific source.

        operator : typing.Optional[str]
            Retrieve only scores with <operator> value.

        value : typing.Optional[float]
            Retrieve only scores with <operator> value.

        score_ids : typing.Optional[str]
            Comma-separated list of score IDs to limit the results to.

        config_id : typing.Optional[str]
            Retrieve only scores with a specific configId.

        queue_id : typing.Optional[str]
            Retrieve only scores with a specific annotation queueId.

        data_type : typing.Optional[ScoreDataType]
            Retrieve only scores with a specific dataType.

        trace_tags : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            Only scores linked to traces that include all of these tags will be returned.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        GetScoresResponse

        Examples
        --------
        import asyncio

        from langfuse.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.score_v_2.get()


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "api/public/v2/scores",
            method="GET",
            params={
                "page": page,
                "limit": limit,
                "userId": user_id,
                "name": name,
                "fromTimestamp": serialize_datetime(from_timestamp)
                if from_timestamp is not None
                else None,
                "toTimestamp": serialize_datetime(to_timestamp)
                if to_timestamp is not None
                else None,
                "environment": environment,
                "source": source,
                "operator": operator,
                "value": value,
                "scoreIds": score_ids,
                "configId": config_id,
                "queueId": queue_id,
                "dataType": data_type,
                "traceTags": trace_tags,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(GetScoresResponse, _response.json())  # type: ignore
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

    async def get_by_id(
        self, score_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> Score:
        """
        Get a score (supports both trace and session scores)

        Parameters
        ----------
        score_id : str
            The unique langfuse identifier of a score

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Score

        Examples
        --------
        import asyncio

        from langfuse.client import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.score_v_2.get_by_id(
                score_id="scoreId",
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"api/public/v2/scores/{jsonable_encoder(score_id)}",
            method="GET",
            request_options=request_options,
        )
        try:
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
                raise NotFoundError(
                    pydantic_v1.parse_obj_as(typing.Any, _response.json())
                )  # type: ignore
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
