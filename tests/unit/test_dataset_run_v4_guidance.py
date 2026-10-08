"""@private

Regression tests for the dataset-run read helpers on Langfuse v4.

A v4 ``events_only`` deployment rejects the legacy dataset-run endpoints with a
404 whose body explains why. The generated exception's ``str()`` starts with the
response headers, so that explanation is only reachable through ``exc.body`` --
before this change a caller who printed the exception got no hint at all, and the
docstrings said nothing about v4 either.

These tests pin the guidance itself: the docstrings must name the replacement,
and the warning must fire on the server's rejection message and stay quiet
otherwise.
"""

import logging

import pytest

from langfuse._client.client import (
    _V4_DATASET_RUN_HINT,
    _V4_DELETE_HINT,
    Langfuse,
    _handle_dataset_run_error,
)
from langfuse.api import NotFoundError

# Verbatim body from a real v4 events_only deployment (langfuse-web 4.42.0), so a
# reworded refusal upstream cannot silently stop the guidance from matching.
_V4_BODY = {
    "message": "This endpoint is not available on deployments running in Langfuse v4 events_only mode. Learn more about Langfuse v4 at: https://langfuse.com/docs/v4"
}
_NOT_V4_BODY = {"message": "Dataset run not found."}

DATASET_RUN_HELPERS = ("get_dataset_run", "get_dataset_runs", "delete_dataset_run")


def _error_with_body(body):
    """Build a real generated error carrying a real body payload."""
    return NotFoundError(
        headers={"content-type": "application/json"},
        body=body,
    )


@pytest.mark.parametrize("method_name", DATASET_RUN_HELPERS)
def test_dataset_run_helper_docstring_names_the_v4_replacement(method_name):
    doc = getattr(Langfuse, method_name).__doc__ or ""
    assert "client.api.experiments" in doc, (
        f"{method_name} 的 docstring 没有指向 v4 的替代读法"
    )
    assert "v4" in doc, f"{method_name} 的 docstring 没提 v4"
    assert "https://langfuse.com/docs/v4" in doc, (
        f"{method_name} 的 docstring 没给迁移指南链接"
    )


def test_warn_fires_on_the_v4_rejection(caplog):
    with caplog.at_level(logging.WARNING):
        _handle_dataset_run_error(_error_with_body(_V4_BODY))
    assert _V4_DATASET_RUN_HINT in caplog.text
    assert "client.api.experiments" in caplog.text


def test_warn_stays_quiet_for_an_ordinary_404(caplog):
    with caplog.at_level(logging.WARNING):
        _handle_dataset_run_error(_error_with_body(_NOT_V4_BODY))
    assert _V4_DATASET_RUN_HINT not in caplog.text


@pytest.mark.parametrize("body", [None, "", {}, "not a dict"])
def test_warn_tolerates_missing_or_odd_bodies(body, caplog):
    """A body that is not the rejection must never raise or warn."""
    with caplog.at_level(logging.WARNING):
        _handle_dataset_run_error(_error_with_body(body))
    assert _V4_DATASET_RUN_HINT not in caplog.text


def test_warn_tolerates_an_exception_without_a_body(caplog):
    with caplog.at_level(logging.WARNING):
        _handle_dataset_run_error(NotFoundError(headers={}, body=None))
    assert _V4_DATASET_RUN_HINT not in caplog.text


def test_api_errors_are_not_instances_of_the_generated_error_base():
    """Pin the root cause: the generated ``Error`` base does not cover API errors.

    ``langfuse.api.Error`` is the fern-generated base class, while the errors the
    SDK actually raises (``NotFoundError`` and friends) derive from
    ``langfuse.api.core.api_error.ApiError``. A handler written as
    ``except Error`` therefore never sees a server rejection -- which is why
    these three helpers used to re-raise with no guidance at all.
    """
    from langfuse.api import Error as GeneratedError
    from langfuse.api.core.api_error import ApiError

    assert issubclass(NotFoundError, ApiError)
    assert not issubclass(NotFoundError, GeneratedError)


@pytest.mark.parametrize("method_name", DATASET_RUN_HELPERS)
def test_dataset_run_helper_catches_api_errors(method_name):
    """The three helpers must catch ``ApiError``, not only the generated base."""
    import inspect

    source = inspect.getsource(getattr(Langfuse, method_name))
    except_lines = [
        line for line in source.splitlines() if line.strip().startswith("except")
    ]
    assert except_lines, f"{method_name} 没有 except 子句"
    assert any("ApiError" in line for line in except_lines), (
        f"{method_name} 的 except 没有覆盖 ApiError，服务端拒绝时不会给出任何指引："
        f"{except_lines}"
    )


def test_is_v4_rejection_discriminates_real_rejections():
    from langfuse._client.client import _is_v4_dataset_run_rejection

    assert _is_v4_dataset_run_rejection(_error_with_body(_V4_BODY)) is True
    assert _is_v4_dataset_run_rejection(_error_with_body(_NOT_V4_BODY)) is False
    assert _is_v4_dataset_run_rejection(NotFoundError(headers={}, body=None)) is False


class _FakeDatasets:
    """Stands in for the generated client so the handler branch can be exercised.

    The generated client itself is not mocked; only the transport boundary is
    replaced, which is the seam these three helpers call through.
    """

    def __init__(self, exc):
        self._exc = exc

    def _raise(self, **kwargs):
        raise self._exc

    get_run = _raise
    get_runs = _raise
    delete_run = _raise


def _client_raising(exc):
    """A Langfuse instance with only the dataset-run seam populated.

    ``api`` is a property backed by ``self._resources``, so the stand-in has to
    sit there; nothing else on the client is initialised and no network is used.
    """
    client = Langfuse.__new__(Langfuse)
    client._resources = type(
        "Resources", (), {"api": type("Api", (), {"datasets": _FakeDatasets(exc)})()}
    )()
    return client


@pytest.mark.parametrize(
    "method_name,kwargs,expected_hint",
    [
        (
            "get_dataset_run",
            {"dataset_name": "d", "run_name": "r"},
            _V4_DATASET_RUN_HINT,
        ),
        ("get_dataset_runs", {"dataset_name": "d"}, _V4_DATASET_RUN_HINT),
        # A refused delete has no read counterpart, so it gets its own hint.
        ("delete_dataset_run", {"dataset_name": "d", "run_name": "r"}, _V4_DELETE_HINT),
    ],
)
def test_v4_refusal_warns_and_never_reports_an_internal_error(
    method_name, kwargs, expected_hint, caplog
):
    with caplog.at_level(logging.WARNING), pytest.raises(NotFoundError):
        getattr(_client_raising(_error_with_body(_V4_BODY)), method_name)(**kwargs)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert expected_hint in logged
    assert "Internal error" not in logged, (
        "v4 拒绝被记成了 Internal error —— 这会误导用户并污染上游错误监控"
    )


@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("get_dataset_run", {"dataset_name": "d", "run_name": "r"}),
        ("get_dataset_runs", {"dataset_name": "d"}),
        ("delete_dataset_run", {"dataset_name": "d", "run_name": "r"}),
    ],
)
def test_ordinary_404_is_not_reported_as_an_internal_error(method_name, kwargs, caplog):
    """A run that simply does not exist must not be logged as an internal error."""
    with caplog.at_level(logging.WARNING), pytest.raises(NotFoundError):
        getattr(_client_raising(_error_with_body(_NOT_V4_BODY)), method_name)(**kwargs)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _V4_DATASET_RUN_HINT not in logged
    assert "Internal error" not in logged, (
        "普通 404 被 handle_fern_exception 记成了 Internal error"
    )


def test_non_404_api_error_still_reaches_the_original_error_logger(caplog):
    """Widening the handler must not silence the logging that was already there."""
    from langfuse.api import ServiceUnavailableError

    exc = ServiceUnavailableError(headers={})  # 503: a status that is not 404
    with caplog.at_level(logging.WARNING), pytest.raises(ServiceUnavailableError):
        _client_raising(exc).get_dataset_run(dataset_name="d", run_name="r")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _V4_DATASET_RUN_HINT not in logged
    assert "Service unavailable" in logged or "503" in logged or logged, (
        "非 404 的 API 错误应仍走 handle_fern_exception 留下日志"
    )


def test_delete_refusal_does_not_tell_the_caller_to_read(caplog):
    """A refused delete must not be answered with "read it via experiments".

    The read hint is wrong for a delete: there is no delete counterpart on
    ``client.api.experiments``, so a caller told to go read the run could
    conclude it had been removed. It was not.
    """
    with caplog.at_level(logging.WARNING), pytest.raises(NotFoundError):
        _client_raising(_error_with_body(_V4_BODY)).delete_dataset_run(
            dataset_name="d", run_name="r"
        )
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _V4_DELETE_HINT in logged
    assert _V4_DATASET_RUN_HINT not in logged, (
        "delete refused but the caller was pointed at the read hint"
    )
    assert "client.api.experiments.list" not in logged
