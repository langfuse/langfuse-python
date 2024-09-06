import logging
from langfuse.request import APIErrors, APIError
from langfuse.parse_error import (
    generate_error_message,
    generate_error_messge_fern,
    handle_exception,
)
from langfuse.api.resources.commons.errors import (
    AccessDeniedError,
    MethodNotAllowedError,
    NotFoundError,
    UnauthorizedError,
)
from langfuse.api.core import ApiError
from langfuse.api.resources.health.errors import ServiceUnavailableError


def test_generate_error_message_api_error():
    exception = APIError(message="Test API error", status="500")
    expected_message = "API error occurred: Internal server error occurred. For help, please contact support: https://langfuse.com/support"
    assert expected_message in generate_error_message(exception)


def test_generate_error_message_api_errors():
    errors = [
        APIError(status=400, message="Bad request", details="Invalid input"),
        APIError(status=401, message="Unauthorized", details="Invalid credentials"),
    ]
    exception = APIErrors(errors)
    expected_message = (
        "API errors occurred: "
        "Bad request. Please check your request for any missing or incorrect parameters. Refer to our API docs: https://api.reference.langfuse.com for details.\n"
        "Unauthorized. Please check your public/private host settings. Refer to our installation and setup guide: https://langfuse.com/docs/sdk/typescript/guide for details on SDK configuration."
    )
    assert expected_message in generate_error_message(exception)


def test_generate_error_message_generic_exception():
    exception = Exception("Generic error")
    expected_message = "Unexpected error occurred. Please check your request and contact support: https://langfuse.com/support."
    assert generate_error_message(exception) == expected_message


def test_generate_error_message_access_denied_error():
    exception = AccessDeniedError(body={})
    expected_message = "Forbidden. Please check your access control settings. Refer to our RBAC docs: https://langfuse.com/docs/rbac for details."
    assert generate_error_messge_fern(exception) == expected_message


def test_generate_error_message_method_not_allowed_error():
    exception = MethodNotAllowedError(body={})
    expected_message = "Unexpected error occurred. Please check your request and contact support: https://langfuse.com/support."
    assert generate_error_messge_fern(exception) == expected_message


def test_generate_error_message_not_found_error():
    exception = NotFoundError(body={})
    expected_message = "Internal error occurred. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_messge_fern(exception) == expected_message


def test_generate_error_message_unauthorized_error():
    exception = UnauthorizedError(body={})
    expected_message = "Unauthorized. Please check your public/private host settings. Refer to our installation and setup guide: https://langfuse.com/docs/sdk/typescript/guide for details on SDK configuration."
    assert generate_error_messge_fern(exception) == expected_message


def test_generate_error_message_service_unavailable_error():
    exception = ServiceUnavailableError()
    expected_message = "Service unavailable. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_messge_fern(exception) == expected_message


def test_generate_error_message_generic():
    exception = ApiError(status_code=503)
    expected_message = "Service unavailable. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_messge_fern(exception) == expected_message
