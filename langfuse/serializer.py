from datetime import date, datetime, timezone
import json
from typing import Any


class DatetimeSerializer(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, (date, datetime)):
            # make the datetime object timezone-aware if it isn't already
            if obj.tzinfo is None or obj.tzinfo.utcoffset(obj) is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()

        return json.JSONEncoder.default(self, obj)
