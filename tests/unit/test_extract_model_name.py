"""Unit tests for ``langfuse.langchain.utils._extract_model_name``."""

from langfuse.langchain.utils import _extract_model_name


def _azure_openai_serialized(
    *, deployment_name: str, openai_api_version: str | None
) -> dict:
    """Build a minimal ``serialized`` dict mirroring what LangChain emits for AzureOpenAI."""
    kwargs: dict = {"deployment_name": deployment_name}
    if openai_api_version is not None:
        kwargs["openai_api_version"] = openai_api_version
    return {"id": ["langchain", "llms", "openai", "AzureOpenAI"], "kwargs": kwargs}


def test_azure_openai_returns_deployment_name_with_api_version_suffix():
    """Regression: ``openai_api_version`` should be appended to the deployment name.

    Previously the code read the value from a non-existent ``deployment_version``
    key after checking that ``openai_api_version`` was set, so the suffix was
    silently dropped.
    """
    serialized = _azure_openai_serialized(
        deployment_name="my-deployment",
        openai_api_version="2024-02-15-preview",
    )
    assert _extract_model_name(serialized) == "my-deployment-2024-02-15-preview"


def test_azure_openai_returns_bare_deployment_name_when_no_api_version():
    serialized = _azure_openai_serialized(
        deployment_name="my-deployment",
        openai_api_version=None,
    )
    assert _extract_model_name(serialized) == "my-deployment"


def test_azure_openai_no_duplicate_version_suffix():
    """If the deployment_name already contains the version, don't append it again.

    This guards the existing behavior introduced in PR #1203.
    """
    serialized = _azure_openai_serialized(
        deployment_name="my-deployment-2024-02-15-preview",
        openai_api_version="2024-02-15-preview",
    )
    assert _extract_model_name(serialized) == "my-deployment-2024-02-15-preview"
