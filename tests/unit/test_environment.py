import pytest

from langfuse._utils.environment import common_release_envs, get_common_release_envs


@pytest.mark.parametrize(
    "values,expected",
    [
        ({}, None),
        ({name: "" for name in common_release_envs}, None),
        ({"RENDER_GIT_COMMIT": "", "GITHUB_SHA": "github-sha"}, "github-sha"),
        (
            {"RENDER_GIT_COMMIT": "render-sha", "GITHUB_SHA": "github-sha"},
            "render-sha",
        ),
        ({"RENDER_GIT_COMMIT": " ", "GITHUB_SHA": "github-sha"}, " "),
    ],
    ids=["unset", "all-empty", "skip-empty", "precedence", "preserve-whitespace"],
)
def test_get_common_release_envs(monkeypatch, values, expected):
    for name in common_release_envs:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert get_common_release_envs() == expected
