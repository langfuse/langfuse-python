# This file was auto-generated by Fern from our API Definition.

import typing
from ...core.client_wrapper import SyncClientWrapper
from ..commons.types.dataset_status import DatasetStatus
from ...core.request_options import RequestOptions
from ..commons.types.dataset_item import DatasetItem
from ...core.pydantic_utilities import parse_obj_as
from ..commons.errors.error import Error
from ..commons.errors.unauthorized_error import UnauthorizedError
from ..commons.errors.access_denied_error import AccessDeniedError
from ..commons.errors.method_not_allowed_error import MethodNotAllowedError
from ..commons.errors.not_found_error import NotFoundError
from json.decoder import JSONDecodeError
from ...core.api_error import ApiError
from ...core.jsonable_encoder import jsonable_encoder
from .types.paginated_dataset_items import PaginatedDatasetItems
from ...core.client_wrapper import AsyncClientWrapper

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class DatasetItemsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def create(
        self,
        *,
        dataset_name: str,
        input: typing.Optional[typing.Optional[typing.Any]] = OMIT,
        expected_output: typing.Optional[typing.Optional[typing.Any]] = OMIT,
        metadata: typing.Optional[typing.Optional[typing.Any]] = OMIT,
        source_trace_id: typing.Optional[str] = OMIT,
        source_observation_id: typing.Optional[str] = OMIT,
        id: typing.Optional[str] = OMIT,
        status: typing.Optional[DatasetStatus] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> DatasetItem:
        """
        Create a dataset item

        Parameters
        ----------
        dataset_name : str

        input : typing.Optional[typing.Optional[typing.Any]]

        expected_output : typing.Optional[typing.Optional[typing.Any]]

        metadata : typing.Optional[typing.Optional[typing.Any]]

        source_trace_id : typing.Optional[str]

        source_observation_id : typing.Optional[str]

        id : typing.Optional[str]
            Dataset items are upserted on their id. Id needs to be unique (project-level) and cannot be reused across datasets.

        status : typing.Optional[DatasetStatus]
            Defaults to ACTIVE for newly created items

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        DatasetItem

        Examples
        --------
        from finto import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.dataset_items.create(
            dataset_name="string",
            input={"key": "value"},
            expected_output={"key": "value"},
            metadata={"key": "value"},
            source_trace_id="string",
            source_observation_id="string",
            id="string",
            status="ACTIVE",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "api/public/dataset-items",
            method="POST",
            json={
                "datasetName": dataset_name,
                "input": input,
                "expectedOutput": expected_output,
                "metadata": metadata,
                "sourceTraceId": source_trace_id,
                "sourceObservationId": source_observation_id,
                "id": id,
                "status": status,
            },
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    DatasetItem,
                    parse_obj_as(
                        type_=DatasetItem,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 400:
                raise Error(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 401:
                raise UnauthorizedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 403:
                raise AccessDeniedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def get(
        self, id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> DatasetItem:
        """
        Get a dataset item

        Parameters
        ----------
        id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        DatasetItem

        Examples
        --------
        from finto import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.dataset_items.get(
            id="string",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            f"api/public/dataset-items/{jsonable_encoder(id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    DatasetItem,
                    parse_obj_as(
                        type_=DatasetItem,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 400:
                raise Error(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 401:
                raise UnauthorizedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 403:
                raise AccessDeniedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def list(
        self,
        *,
        dataset_name: typing.Optional[str] = None,
        source_trace_id: typing.Optional[str] = None,
        source_observation_id: typing.Optional[str] = None,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> PaginatedDatasetItems:
        """
        Get dataset items

        Parameters
        ----------
        dataset_name : typing.Optional[str]

        source_trace_id : typing.Optional[str]

        source_observation_id : typing.Optional[str]

        page : typing.Optional[int]
            page number, starts at 1

        limit : typing.Optional[int]
            limit of items per page

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        PaginatedDatasetItems

        Examples
        --------
        from finto import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.dataset_items.list(
            dataset_name="string",
            source_trace_id="string",
            source_observation_id="string",
            page=1,
            limit=1,
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "api/public/dataset-items",
            method="GET",
            params={
                "datasetName": dataset_name,
                "sourceTraceId": source_trace_id,
                "sourceObservationId": source_observation_id,
                "page": page,
                "limit": limit,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    PaginatedDatasetItems,
                    parse_obj_as(
                        type_=PaginatedDatasetItems,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 400:
                raise Error(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 401:
                raise UnauthorizedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 403:
                raise AccessDeniedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncDatasetItemsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def create(
        self,
        *,
        dataset_name: str,
        input: typing.Optional[typing.Optional[typing.Any]] = OMIT,
        expected_output: typing.Optional[typing.Optional[typing.Any]] = OMIT,
        metadata: typing.Optional[typing.Optional[typing.Any]] = OMIT,
        source_trace_id: typing.Optional[str] = OMIT,
        source_observation_id: typing.Optional[str] = OMIT,
        id: typing.Optional[str] = OMIT,
        status: typing.Optional[DatasetStatus] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> DatasetItem:
        """
        Create a dataset item

        Parameters
        ----------
        dataset_name : str

        input : typing.Optional[typing.Optional[typing.Any]]

        expected_output : typing.Optional[typing.Optional[typing.Any]]

        metadata : typing.Optional[typing.Optional[typing.Any]]

        source_trace_id : typing.Optional[str]

        source_observation_id : typing.Optional[str]

        id : typing.Optional[str]
            Dataset items are upserted on their id. Id needs to be unique (project-level) and cannot be reused across datasets.

        status : typing.Optional[DatasetStatus]
            Defaults to ACTIVE for newly created items

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        DatasetItem

        Examples
        --------
        import asyncio

        from finto import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.dataset_items.create(
                dataset_name="string",
                input={"key": "value"},
                expected_output={"key": "value"},
                metadata={"key": "value"},
                source_trace_id="string",
                source_observation_id="string",
                id="string",
                status="ACTIVE",
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "api/public/dataset-items",
            method="POST",
            json={
                "datasetName": dataset_name,
                "input": input,
                "expectedOutput": expected_output,
                "metadata": metadata,
                "sourceTraceId": source_trace_id,
                "sourceObservationId": source_observation_id,
                "id": id,
                "status": status,
            },
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    DatasetItem,
                    parse_obj_as(
                        type_=DatasetItem,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 400:
                raise Error(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 401:
                raise UnauthorizedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 403:
                raise AccessDeniedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def get(
        self, id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> DatasetItem:
        """
        Get a dataset item

        Parameters
        ----------
        id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        DatasetItem

        Examples
        --------
        import asyncio

        from finto import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.dataset_items.get(
                id="string",
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"api/public/dataset-items/{jsonable_encoder(id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    DatasetItem,
                    parse_obj_as(
                        type_=DatasetItem,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 400:
                raise Error(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 401:
                raise UnauthorizedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 403:
                raise AccessDeniedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def list(
        self,
        *,
        dataset_name: typing.Optional[str] = None,
        source_trace_id: typing.Optional[str] = None,
        source_observation_id: typing.Optional[str] = None,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> PaginatedDatasetItems:
        """
        Get dataset items

        Parameters
        ----------
        dataset_name : typing.Optional[str]

        source_trace_id : typing.Optional[str]

        source_observation_id : typing.Optional[str]

        page : typing.Optional[int]
            page number, starts at 1

        limit : typing.Optional[int]
            limit of items per page

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        PaginatedDatasetItems

        Examples
        --------
        import asyncio

        from finto import AsyncFernLangfuse

        client = AsyncFernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.dataset_items.list(
                dataset_name="string",
                source_trace_id="string",
                source_observation_id="string",
                page=1,
                limit=1,
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "api/public/dataset-items",
            method="GET",
            params={
                "datasetName": dataset_name,
                "sourceTraceId": source_trace_id,
                "sourceObservationId": source_observation_id,
                "page": page,
                "limit": limit,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    PaginatedDatasetItems,
                    parse_obj_as(
                        type_=PaginatedDatasetItems,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 400:
                raise Error(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 401:
                raise UnauthorizedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 403:
                raise AccessDeniedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 405:
                raise MethodNotAllowedError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
