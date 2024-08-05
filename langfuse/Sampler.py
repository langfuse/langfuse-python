import random


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
            else:
                raise Exception("Unexpected event properties")

            return self.deterministic_sample(trace_id, self.sample_rate)

        else:
            raise Exception("Unexpected event properties")

    def deterministic_sample(self, trace_id: str, sample_rate: float):
        hash_value = hash(trace_id)
        normalized_hash = (hash_value & 0xFFFFFFFF) / 2**32  # Normalize to [0, 1)
        return normalized_hash < sample_rate
