import logging
from typing import Union

from openai import APIError

from langfuse.request import APIErrors

SUPPORT_URL = "https://langfuse.com/support"
API_DOCS_URL = "https://api.reference.langfuse.com"
RBAC_DOCS_URL = "https://langfuse.com/docs/rbac"
RATE_LIMITS_URL = "https://langfuse.com/faq/all/api-limits"
NPM_PACKAGE_URL = "https://pypi.org/project/langfuse/"

# Error messages
updatePromptResponse = (
    f"Make sure to keep your SDK updated, refer to {NPM_PACKAGE_URL} for details."
)
defaultErrorResponse = f"Unexpected error occurred. Please check your request and contact support: {SUPPORT_URL}."

# Error response map
errorResponseByCode = {
    "500": f"Internal server error occurred. Please contact support: {SUPPORT_URL}",
    "501": f"Not implemented. Please check your request and contact support: {SUPPORT_URL}",
    "502": f"Bad gateway. Please try again later and contact support: {SUPPORT_URL}.",
    "503": f"Service unavailable. Please try again later and contact support if the error persists: {SUPPORT_URL}.",
    "504": "Gateway timeout. Please try again later and contact support: {SUPPORT_URL}.",
    "404": f"Internal error occurred. Likely caused by race condition, please escalate to support if seen in high volume: {SUPPORT_URL}",
    "400": f"Bad request. Please check your request for any missing or incorrect parameters. Refer to our API docs: {API_DOCS_URL} for details.",
    "401": "Unauthorized. Please check your public/private host settings.",
    "403": f"Forbidden. Please check your access control settings. Refer to our RBAC docs: {RBAC_DOCS_URL} for details.",
    "429": f"Rate limit exceeded. Please try again later. For more information on rate limits please see: {RATE_LIMITS_URL}",
}


def handle_exception(exception: Union[APIError, APIErrors, Exception]) -> None:
    log = logging.getLogger("langfuse")

    log.debug(exception)

    if isinstance(exception, APIError):
        error_message = f"API error occurred: {errorResponseByCode.get(exception.status_code, defaultErrorResponse)}"
        log.error(error_message)
    elif isinstance(exception, APIErrors):
        error_messages = [
            errorResponseByCode.get(error.status, defaultErrorResponse)
            for error in exception.errors
        ]
        combined_error_message = "API errors occurred: " + "\n".join(error_messages)
        log.error(combined_error_message)
    elif isinstance(exception, Exception):
        log.error(defaultErrorResponse)
    else:
        log.error(defaultErrorResponse)
