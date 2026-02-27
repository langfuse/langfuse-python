"""Test suite for datetime_utils module - UTC timestamp serialization."""

import datetime as dt
from datetime import timezone

import pytest

from langfuse.api.core.datetime_utils import serialize_datetime


class TestSerializeDatetime:
    """Test suite for the serialize_datetime function."""

    def test_utc_datetime_ends_with_z(self):
        """Test that UTC datetime is serialized with 'Z' suffix."""
        utc_dt = dt.datetime(2025, 12, 10, 13, 30, 45, 123456, tzinfo=timezone.utc)
        result = serialize_datetime(utc_dt)
        
        assert result.endswith("Z")
        assert "+00:00" not in result
        assert result == "2025-12-10T13:30:45.123456Z"

    def test_utc_datetime_without_microseconds(self):
        """Test UTC datetime without microseconds."""
        utc_dt = dt.datetime(2025, 12, 10, 13, 30, 45, tzinfo=timezone.utc)
        result = serialize_datetime(utc_dt)
        
        assert result.endswith("Z")
        assert result == "2025-12-10T13:30:45Z"

    def test_naive_datetime_assumed_utc(self):
        """Test that naive datetime (no tzinfo) is assumed to be UTC.
        
        This is the key fix: naive datetime should be treated as UTC,
        not local time, to prevent duplicate trace records in ClickHouse
        when the SDK runs in non-UTC timezones.
        """
        naive_dt = dt.datetime(2025, 12, 10, 13, 30, 45, 123456)
        result = serialize_datetime(naive_dt)
        
        # Should end with 'Z' (UTC), not a local timezone offset like +08:00
        assert result.endswith("Z"), f"Expected UTC suffix 'Z', got: {result}"
        assert result == "2025-12-10T13:30:45.123456Z"

    def test_naive_datetime_without_microseconds(self):
        """Test naive datetime without microseconds is assumed UTC."""
        naive_dt = dt.datetime(2025, 12, 10, 13, 30, 45)
        result = serialize_datetime(naive_dt)
        
        assert result.endswith("Z")
        assert result == "2025-12-10T13:30:45Z"

    def test_non_utc_timezone_uses_offset(self):
        """Test that non-UTC timezones use offset format."""
        # Create datetime with +08:00 timezone
        tz_plus_8 = timezone(dt.timedelta(hours=8))
        dt_plus_8 = dt.datetime(2025, 12, 10, 21, 30, 45, tzinfo=tz_plus_8)
        result = serialize_datetime(dt_plus_8)
        
        # Should use offset format, not 'Z'
        assert result.endswith("+08:00")
        assert result == "2025-12-10T21:30:45+08:00"

    def test_negative_timezone_offset(self):
        """Test negative timezone offset format."""
        tz_minus_5 = timezone(dt.timedelta(hours=-5))
        dt_minus_5 = dt.datetime(2025, 12, 10, 8, 30, 45, tzinfo=tz_minus_5)
        result = serialize_datetime(dt_minus_5)
        
        assert result.endswith("-05:00")
        assert result == "2025-12-10T08:30:45-05:00"

    def test_consistency_with_internal_timestamp_function(self):
        """Test that serialize_datetime is consistent with _get_timestamp.
        
        The _get_timestamp function returns datetime.now(timezone.utc),
        which should serialize correctly with 'Z' suffix.
        """
        from langfuse._utils import _get_timestamp
        
        timestamp = _get_timestamp()
        result = serialize_datetime(timestamp)
        
        # Should always end with 'Z' since _get_timestamp uses UTC
        assert result.endswith("Z"), f"Expected UTC suffix 'Z', got: {result}"

    def test_multiple_naive_datetimes_serialize_consistently(self):
        """Test that multiple naive datetimes serialize consistently.
        
        This prevents the issue where different events in the same trace
        could get different timezone treatments.
        """
        dt1 = dt.datetime(2025, 12, 10, 13, 30, 45)
        dt2 = dt.datetime(2025, 12, 10, 13, 30, 46)
        dt3 = dt.datetime(2025, 12, 10, 13, 30, 47)
        
        results = [serialize_datetime(d) for d in [dt1, dt2, dt3]]
        
        # All should have 'Z' suffix (UTC)
        for result in results:
            assert result.endswith("Z"), f"Expected UTC suffix 'Z', got: {result}"
        
        # All should have the same date (no timezone shift causing date change)
        for result in results:
            assert result.startswith("2025-12-10")

    def test_edge_case_midnight_utc(self):
        """Test midnight UTC serialization."""
        midnight = dt.datetime(2025, 12, 10, 0, 0, 0, tzinfo=timezone.utc)
        result = serialize_datetime(midnight)
        
        assert result == "2025-12-10T00:00:00Z"

    def test_edge_case_end_of_day_utc(self):
        """Test end of day UTC serialization."""
        end_of_day = dt.datetime(2025, 12, 10, 23, 59, 59, 999999, tzinfo=timezone.utc)
        result = serialize_datetime(end_of_day)
        
        assert result == "2025-12-10T23:59:59.999999Z"

    def test_iso8601_format_compliance(self):
        """Test that output complies with ISO 8601 format."""
        naive_dt = dt.datetime(2025, 12, 10, 13, 30, 45, 123456)
        result = serialize_datetime(naive_dt)
        
        # ISO 8601 format: YYYY-MM-DDTHH:MM:SS.ffffff[Z|+HH:MM|-HH:MM]
        assert "T" in result
        assert result.count(":") >= 2
        # Should be parseable
        parsed = dt.datetime.fromisoformat(result.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

