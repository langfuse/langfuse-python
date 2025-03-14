# This file was auto-generated by Fern from our API Definition.

import typing
from json.decoder import JSONDecodeError

from ...core.api_error import ApiError
from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ...core.jsonable_encoder import jsonable_encoder
from ...core.pydantic_utilities import pydantic_v1
from ...core.request_options import RequestOptions
from ..commons.errors.access_denied_error import AccessDeniedError
from ..commons.errors.error import Error
from ..commons.errors.method_not_allowed_error import MethodNotAllowedError
from ..commons.errors.not_found_error import NotFoundError
from ..commons.errors.unauthorized_error import UnauthorizedError
from ..commons.types.comment import Comment
from .types.create_comment_request import CreateCommentRequest
from .types.create_comment_response import CreateCommentResponse
from .types.get_comments_response import GetCommentsResponse

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class CommentsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def create(
        self,
        *,
        request: CreateCommentRequest,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> CreateCommentResponse:
        """
        Create a comment. Comments may be attached to different object types (trace, observation, session, prompt).

        Parameters
        ----------
        request : CreateCommentRequest

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        CreateCommentResponse

        Examples
        --------
        from langfuse import CreateCommentRequest
        from langfuse.client import FernLangfuse

        client = FernLangfuse(
            x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
            x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
            x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            base_url="https://yourhost.com/path/to/api",
        )
        client.comments.create(
            request=CreateCommentRequest(
                project_id="projectId",
                object_type="objectType",
                object_id="objectId",
                content="content",
            ),
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            "api/public/comments",
            method="POST",
            json=request,
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(CreateCommentResponse, _response.json())  # type: ignore
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

    def get(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        object_type: typing.Optional[str] = None,
        object_id: typing.Optional[str] = None,
        author_user_id: typing.Optional[str] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> GetCommentsResponse:
        """
        Get all comments

        Parameters
        ----------
        page : typing.Optional[int]
            Page number, starts at 1.

        limit : typing.Optional[int]
            Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit

        object_type : typing.Optional[str]
            Filter comments by object type (trace, observation, session, prompt).

        object_id : typing.Optional[str]
            Filter comments by object id. If objectType is not provided, an error will be thrown.

        author_user_id : typing.Optional[str]
            Filter comments by author user id.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        GetCommentsResponse

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
        client.comments.get()
        """
        _response = self._client_wrapper.httpx_client.request(
            "api/public/comments",
            method="GET",
            params={
                "page": page,
                "limit": limit,
                "objectType": object_type,
                "objectId": object_id,
                "authorUserId": author_user_id,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(GetCommentsResponse, _response.json())  # type: ignore
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
        self,
        comment_id: str,
        *,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Comment:
        """
        Get a comment by id

        Parameters
        ----------
        comment_id : str
            The unique langfuse identifier of a comment

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Comment

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
        client.comments.get_by_id(
            comment_id="commentId",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            f"api/public/comments/{jsonable_encoder(comment_id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(Comment, _response.json())  # type: ignore
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


class AsyncCommentsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def create(
        self,
        *,
        request: CreateCommentRequest,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> CreateCommentResponse:
        """
        Create a comment. Comments may be attached to different object types (trace, observation, session, prompt).

        Parameters
        ----------
        request : CreateCommentRequest

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        CreateCommentResponse

        Examples
        --------
        import asyncio

        from langfuse import CreateCommentRequest
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
            await client.comments.create(
                request=CreateCommentRequest(
                    project_id="projectId",
                    object_type="objectType",
                    object_id="objectId",
                    content="content",
                ),
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "api/public/comments",
            method="POST",
            json=request,
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(CreateCommentResponse, _response.json())  # type: ignore
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

    async def get(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        object_type: typing.Optional[str] = None,
        object_id: typing.Optional[str] = None,
        author_user_id: typing.Optional[str] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> GetCommentsResponse:
        """
        Get all comments

        Parameters
        ----------
        page : typing.Optional[int]
            Page number, starts at 1.

        limit : typing.Optional[int]
            Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit

        object_type : typing.Optional[str]
            Filter comments by object type (trace, observation, session, prompt).

        object_id : typing.Optional[str]
            Filter comments by object id. If objectType is not provided, an error will be thrown.

        author_user_id : typing.Optional[str]
            Filter comments by author user id.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        GetCommentsResponse

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
            await client.comments.get()


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "api/public/comments",
            method="GET",
            params={
                "page": page,
                "limit": limit,
                "objectType": object_type,
                "objectId": object_id,
                "authorUserId": author_user_id,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(GetCommentsResponse, _response.json())  # type: ignore
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
        self,
        comment_id: str,
        *,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Comment:
        """
        Get a comment by id

        Parameters
        ----------
        comment_id : str
            The unique langfuse identifier of a comment

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Comment

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
            await client.comments.get_by_id(
                comment_id="commentId",
            )


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"api/public/comments/{jsonable_encoder(comment_id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return pydantic_v1.parse_obj_as(Comment, _response.json())  # type: ignore
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
