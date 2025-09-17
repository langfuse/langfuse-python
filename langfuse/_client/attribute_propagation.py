


from opentelemetry import trace
from typing import Any, Dict, Optional
from opentelemetry import baggage

import opentelemetry.context as otel_context

def propagate_attributes(
    *,
    current_ctx: Optional[otel_context.Context],
    dict_to_propagate: Dict[str, Any],
) -> None:

    """
    Propagate attributes from a dictionary to a span and context.
    """

    ctx = current_ctx or otel_context.get_current()


    for key, value in dict_to_propagate.items():
        print(f"Propagating attribute {key} with value {value}")
        # Baggage values must be strings
        baggage.set_baggage(key, str(value), context=ctx)
    



