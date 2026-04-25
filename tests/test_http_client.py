from langfuse.api.core.http_client import get_request_body


def test_get_request_body_skips_empty_body_when_no_additional_parameters() -> None:
    json_body, data_body = get_request_body(
        json=None,
        data=None,
        request_options={"timeout_in_seconds": 30},
        omit=None,
    )

    assert json_body is None
    assert data_body is None


def test_get_request_body_includes_additional_body_parameters() -> None:
    json_body, data_body = get_request_body(
        json=None,
        data=None,
        request_options={"additional_body_parameters": {"foo": "bar"}},
        omit=None,
    )

    assert json_body == {"foo": "bar"}
    assert data_body is None


def test_get_request_body_skips_empty_additional_body_parameters() -> None:
    json_body, data_body = get_request_body(
        json=None,
        data=None,
        request_options={"additional_body_parameters": {}},
        omit=None,
    )

    assert json_body is None
    assert data_body is None
