import logging
import random

log = logging.getLogger("langfuse")


class Sampler:
    sample_rate: float

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        random.seed(42)  # Fixed seed for reproducibility

    def sample_event(self, event: dict):
        # need to get trace_id from a given event

        if "type" in event and "body" in event:
            event_type = event["type"]

            trace_id = None

            if event_type == "trace-create" and "id" in event["body"]:
                trace_id = event["body"]["id"]
            elif "trace_id" in event["body"]:
                trace_id = event["body"]["trace_id"]
            elif "traceId" in event["body"]:
                trace_id = event["body"]["traceId"]
            else:
                log.error(event)
                raise Exception("No trace id found in event")

            return self.deterministic_sample(trace_id, self.sample_rate)

        else:
            raise Exception("Event has no type or body")

    def deterministic_sample(self, trace_id: str, sample_rate: float):
        log.debug(
            f"Applying deterministic sampling to trace_id: {trace_id} with rate {sample_rate}"
        )
        hash_value = hash(trace_id)
        normalized_hash = (hash_value & 0xFFFFFFFF) / 2**32  # Normalize to [0, 1)

        result = normalized_hash < sample_rate

        if not result:
            log.debug(
                f"event with trace_id: {trace_id} and rate ${sample_rate} was sampled and not sent to the server"
            )

        return result
