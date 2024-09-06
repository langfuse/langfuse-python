import logging
from typing import Union

from openai import APIError

from langfuse.request import APIErrors

SUPPORT_URL = "https://langfuse.com/support"
API_DOCS_URL = "https://api.reference.langfuse.com"
RBAC_DOCS_URL = "https://langfuse.com/docs/rbac"
INSTALLATION_DOCS_URL = "https://langfuse.com/docs/sdk/typescript/guide"
RATE_LIMITS_URL = "https://langfuse.com/faq/all/api-limits"
NPM_PACKAGE_URL = "https://www.npmjs.com/package/langfuse"

# Error messages
updatePromptResponse = (
    f"Make sure to keep your SDK updated, refer to {NPM_PACKAGE_URL} for details."
)
defaultServerErrorPrompt = f"This is an unusual occurrence and we are monitoring it closely. For help, please contact support: {SUPPORT_URL}."
defaultErrorResponse = f"Unexpected error occurred. Please check your request and contact support: {SUPPORT_URL}."

# Error response map
errorResponseByCode = {
    500: f"Internal server error occurred. For help, please contact support: {SUPPORT_URL}",
    501: f"Not implemented. Please check your request and contact support for help: {SUPPORT_URL}.",
    502: f"Bad gateway. {defaultServerErrorPrompt}",
    503: f"Service unavailable. {defaultServerErrorPrompt}",
    504: f"Gateway timeout. {defaultServerErrorPrompt}",
    404: f"Internal error occurred. {defaultServerErrorPrompt}",
    400: f"Bad request. Please check your request for any missing or incorrect parameters. Refer to our API docs: {API_DOCS_URL} for details.",
    401: f"Unauthorized. Please check your public/private host settings. Refer to our installation and setup guide: {INSTALLATION_DOCS_URL} for details on SDK configuration.",
    403: f"Forbidden. Please check your access control settings. Refer to our RBAC docs: {RBAC_DOCS_URL} for details.",
    429: f"Rate limit exceeded. For more information on rate limits please see: {RATE_LIMITS_URL}",
}


def generate_error_message(exception: Union[APIError, APIErrors, Exception]) -> str:
    if isinstance(exception, APIError):
        status_code = (
            int(exception.status)
            if isinstance(exception.status, str)
            else exception.status
        )
        return f"API error occurred: {errorResponseByCode.get(status_code, defaultErrorResponse)}"
    elif isinstance(exception, APIErrors):
        error_messages = [
            errorResponseByCode.get(
                int(error.status) if isinstance(error.status, str) else error.status,
                defaultErrorResponse,
            )
            for error in exception.errors
        ]
        return "API errors occurred: " + "\n".join(error_messages)
    elif isinstance(exception, Exception):
        return defaultErrorResponse
    else:
        return defaultErrorResponse


def handle_exception(exception: Union[APIError, APIErrors, Exception]) -> None:
    log = logging.getLogger("langfuse")
    log.debug(exception)
    error_message = generate_error_message(exception)
    log.error(error_message)
