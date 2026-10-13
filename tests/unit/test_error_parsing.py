"""@private"""

from langfuse._utils.parse_error import generate_error_message_fern
from langfuse.api import (
    AccessDeniedError,
    MethodNotAllowedError,
    NotFoundError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from langfuse.api.core import ApiError


def test_generate_error_message_access_denied_error():
    exception = AccessDeniedError(body={})
    expected_message = "Forbidden. Please check your access control settings. Refer to our RBAC docs: https://langfuse.com/docs/rbac for details."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_method_not_allowed_error():
    exception = MethodNotAllowedError(body={})
    expected_message = "Unexpected error occurred. Please check your request and contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_not_found_error():
    exception = NotFoundError(body={})
    expected_message = "Internal error occurred. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_unauthorized_error():
    exception = UnauthorizedError(body={})
    expected_message = "Unauthorized. Please check your public/private host settings. Refer to our installation and setup guide: https://langfuse.com/docs/sdk/typescript/guide for details on SDK configuration."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_service_unavailable_error():
    exception = ServiceUnavailableError()
    expected_message = "Service unavailable. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_generic():
    exception = ApiError(status_code=503)
    expected_message = "Service unavailable. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message
