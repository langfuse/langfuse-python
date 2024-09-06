import logging
from langfuse.request import APIErrors, APIError
from langfuse.parse_error import generate_error_message, handle_exception


def test_generate_error_message_api_error():
    exception = APIError(message="Test API error", status=500)
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


def test_handle_exception_api_error(caplog):
    exception = APIError(message="Test API error", status=500)
    with caplog.at_level(logging.ERROR):
        handle_exception(exception)
    assert (
        "API error occurred: Internal server error occurred. For help, please contact support: https://langfuse.com/support"
        in caplog.text
    )


def test_handle_exception_api_errors(caplog):
    errors = [
        APIError(status=400, message="Bad request", details="Invalid input"),
        APIError(status=401, message="Unauthorized", details="Invalid credentials"),
    ]
    exception = APIErrors(errors)
    with caplog.at_level(logging.ERROR):
        handle_exception(exception)
    assert "API errors occurred: " in caplog.text
    assert (
        "Bad request. Please check your request for any missing or incorrect parameters. Refer to our API docs: https://api.reference.langfuse.com for details."
        in caplog.text
    )
    assert (
        "Unauthorized. Please check your public/private host settings. Refer to our installation and setup guide: https://langfuse.com/docs/sdk/typescript/guide for details on SDK configuration."
        in caplog.text
    )


def test_handle_exception_generic_exception(caplog):
    exception = Exception("Generic error")
    with caplog.at_level(logging.ERROR):
        handle_exception(exception)
    assert (
        "Unexpected error occurred. Please check your request and contact support: https://langfuse.com/support."
        in caplog.text
    )
