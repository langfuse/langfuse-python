import unittest
from langfuse.Sampler import Sampler


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler(sample_rate=0.5)

    def test_sample_event_trace_create(self):
        event = {"type": "trace-create", "body": {"id": "trace_123"}}
        result = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

        event = {
            "type": "trace-create",
            "body": {"id": "trace_123", "something": "else"},
        }
        result_two = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

        assert result == result_two

    def test_multiple_events_of_different_types(self):
        event = {"type": "trace-create", "body": {"id": "trace_123"}}

        result = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

        event = {"type": "generation-create", "body": {"trace_id": "trace_123"}}
        result_two = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

        event = event = {"type": "score-create", "body": {"trace_id": "trace_123"}}
        result_three = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

        event = {"type": "generation-update", "body": {"traceId": "trace_123"}}
        result_four = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

        assert result == result_two == result_three == result_four

    def test_sample_event_trace_id(self):
        event = {"type": "some-other-type", "body": {"trace_id": "trace_456"}}
        result = self.sampler.sample_event(event)
        self.assertIsInstance(result, bool)

    def test_sample_event_unexpected_properties(self):
        event = {"type": "some-type", "body": {}}
        result = self.sampler.sample_event(event)
        self.assertTrue(result)

    def test_deterministic_sample(self):
        trace_id = "trace_789"
        result = self.sampler.deterministic_sample(trace_id, 0.5)
        self.assertIsInstance(result, bool)

    def test_deterministic_sample_high_rate(self):
        trace_id = "trace_789"
        result = self.sampler.deterministic_sample(trace_id, 1.0)
        self.assertTrue(result)

    def test_deterministic_sample_low_rate(self):
        trace_id = "trace_789"
        result = self.sampler.deterministic_sample(trace_id, 0.0)
        self.assertFalse(result)

    def test_deterministic_sample_50_percent_rate(self):
        trace_ids = [f"trace_{i}" for i in range(1000)]
        sampled_count = sum(
            self.sampler.deterministic_sample(trace_id, 0.5) for trace_id in trace_ids
        )
        print(sampled_count)
        self.assertTrue(
            450 <= sampled_count <= 550,
            f"Sampled count {sampled_count} is not within the expected range",
        )

    def test_deterministic_sample_10_percent_rate(self):
        trace_ids = [f"trace_{i}" for i in range(1000)]
        sampled_count = sum(
            self.sampler.deterministic_sample(trace_id, 0.1) for trace_id in trace_ids
        )
        print(sampled_count)
        self.assertTrue(
            90 <= sampled_count <= 110,
            f"Sampled count {sampled_count} is not within the expected range",
        )
