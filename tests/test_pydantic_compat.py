from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timezone

from langfuse.api.core.pydantic_utilities import parse_date, parse_datetime


def test_import_langfuse_with_user_warnings_as_errors() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", "import langfuse"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_parse_helpers_support_pydantic_v2() -> None:
    assert parse_date("2024-01-02") == date(2024, 1, 2)
    assert parse_datetime("2024-01-02T03:04:05Z") == datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )
