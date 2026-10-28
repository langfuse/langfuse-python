"""The generated skills API can request inline contents without requiring them."""

import httpx
import pytest

from langfuse.api import AsyncLangfuseAPI, LangfuseAPI, SkillVersion


@pytest.fixture
def skill_response():
    return {
        "id": "skill-1",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "projectId": "project-1",
        "createdBy": "user-1",
        "name": "refund-policy",
        "description": "Handle refunds",
        "frontmatter": {},
        "version": 2,
        "tags": [],
        "labels": ["production"],
        "files": [
            {
                "id": "file-1",
                "path": "SKILL.md",
                "blobId": "blob-1",
                "sha256Hash": "hash",
                "contentType": "text/markdown",
                "contentLength": 9,
            }
        ],
    }


def _respond(skill_response, include_content):
    def handle(request):
        assert request.url.path == "/api/public/skills/refund-policy"
        if include_content is None:
            assert "includeContent" not in request.url.params
        else:
            assert request.url.params["includeContent"] == str(include_content).lower()
        if include_content:
            skill_response["files"][0]["content"] = "# Refunds"
        return httpx.Response(200, json=skill_response)

    return handle


@pytest.mark.parametrize("include_content", [None, False, True])
def test_sync_get_skill_inline_content(skill_response, include_content):
    with httpx.Client(
        transport=httpx.MockTransport(_respond(skill_response, include_content))
    ) as http_client:
        api = LangfuseAPI(base_url="https://langfuse.test", httpx_client=http_client)
        skill = api.skills.get("refund-policy", include_content=include_content)

    assert isinstance(skill, SkillVersion)
    assert skill.files[0].content == ("# Refunds" if include_content else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_content", [None, False, True])
async def test_async_get_skill_inline_content(skill_response, include_content):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_respond(skill_response, include_content))
    ) as http_client:
        api = AsyncLangfuseAPI(
            base_url="https://langfuse.test", httpx_client=http_client
        )
        skill = await api.skills.get("refund-policy", include_content=include_content)

    assert isinstance(skill, SkillVersion)
    assert skill.files[0].content == ("# Refunds" if include_content else None)
