from langfuse.api import (
    AccessDeniedError,
    MethodNotAllowedError,
    NotFoundError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from langfuse.api.core import ApiError
from langfuse.logger import langfuse_logger as logger

SUPPORT_URL = "https://langfuse.com/support"
API_DOCS_URL = "https://api.reference.langfuse.com"
RBAC_DOCS_URL = "https://langfuse.com/docs/rbac"
CREDENTIAL_SETUP_URL = (
    "https://langfuse.com/docs/observability/sdk/overview#configure-credentials"
)
RATE_LIMITS_URL = "https://langfuse.com/faq/all/api-limits"
PYPI_PACKAGE_URL = "https://pypi.org/project/langfuse/"

# Error messages
updatePromptResponse = (
    f"Make sure to keep your SDK updated, refer to {PYPI_PACKAGE_URL} for details."
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
    401: f"Unauthorized. Check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL. See the SDK credential setup guide: {CREDENTIAL_SETUP_URL}",
    403: f"Forbidden. Please check your access control settings. Refer to our RBAC docs: {RBAC_DOCS_URL} for details.",
    405: f"Method not allowed. Check LANGFUSE_BASE_URL and whether your server version supports this endpoint. For help, contact support: {SUPPORT_URL}",
    429: f"Rate limit exceeded. For more information on rate limits please see: {RATE_LIMITS_URL}",
}


def generate_error_message_fern(error: ApiError) -> str:
    if isinstance(error, AccessDeniedError):
        return errorResponseByCode.get(403, defaultErrorResponse)
    elif isinstance(error, MethodNotAllowedError):
        return errorResponseByCode.get(405, defaultErrorResponse)
    elif isinstance(error, NotFoundError):
        return errorResponseByCode.get(404, defaultErrorResponse)
    elif isinstance(error, UnauthorizedError):
        return errorResponseByCode.get(401, defaultErrorResponse)
    elif isinstance(error, ServiceUnavailableError):
        return errorResponseByCode.get(503, defaultErrorResponse)
    elif isinstance(error, ApiError):
        status_code = error.status_code
        return (
            errorResponseByCode.get(status_code, defaultErrorResponse)
            if status_code is not None
            else defaultErrorResponse
        )

    return defaultErrorResponse  # type: ignore


def handle_fern_exception(exception: ApiError) -> None:
    logger.debug(exception)
    error_message = generate_error_message_fern(exception)
    logger.error(error_message)
