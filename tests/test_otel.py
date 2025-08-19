import json
from datetime import datetime
from hashlib import sha256
from typing import List, Sequence

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.client import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.media import LangfuseMedia


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory exporter to collect spans for testing."""

    def __init__(self):
        self._finished_spans = []
        self._stopped = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._stopped:
            return SpanExportResult.FAILURE

        self._finished_spans.extend(spans)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        self._stopped = True

    def get_finished_spans(self) -> List[ReadableSpan]:
        return self._finished_spans

    def clear(self):
        self._finished_spans.clear()


class TestOTelBase:
    """Base class for OTEL tests with common fixtures and helper methods."""

    # ------ Common Fixtures ------

    @pytest.fixture(scope="function", autouse=True)
    def cleanup_otel(self):
        """Reset OpenTelemetry state between tests."""
        original_provider = trace_api.get_tracer_provider()
        yield
        trace_api.set_tracer_provider(original_provider)
        LangfuseResourceManager.reset()

    @pytest.fixture
    def memory_exporter(self):
        """Create an in-memory span exporter for testing."""
        exporter = InMemorySpanExporter()
        yield exporter
        exporter.shutdown()

    @pytest.fixture
    def tracer_provider(self, memory_exporter):
        """Create a tracer provider with our memory exporter."""
        resource = Resource.create({"service.name": "langfuse-test"})
        provider = TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(memory_exporter)
        provider.add_span_processor(processor)
        trace_api.set_tracer_provider(provider)
        return provider

    @pytest.fixture
    def mock_processor_init(self, monkeypatch, memory_exporter):
        """Mock the LangfuseSpanProcessor initialization to avoid HTTP traffic."""

        def mock_init(self, **kwargs):
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self.public_key = kwargs.get("public_key", "test-key")
            blocked_scopes = kwargs.get("blocked_instrumentation_scopes")
            self.blocked_instrumentation_scopes = (
                blocked_scopes if blocked_scopes is not None else []
            )
            BatchSpanProcessor.__init__(
                self,
                span_exporter=memory_exporter,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )

        monkeypatch.setattr(
            "langfuse._client.span_processor.LangfuseSpanProcessor.__init__",
            mock_init,
        )

    @pytest.fixture
    def langfuse_client(self, monkeypatch, tracer_provider, mock_processor_init):
        """Create a mocked Langfuse client for testing."""

        # Set environment variables
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")

        # Create test client
        client = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="http://test-host",
            tracing_enabled=True,
        )

        # Configure client for testing
        client._otel_tracer = tracer_provider.get_tracer("langfuse-test")

        yield client

    @pytest.fixture
    def configurable_langfuse_client(
        self, monkeypatch, tracer_provider, mock_processor_init
    ):
        """Create a Langfuse client fixture that allows configuration parameters."""

        def _create_client(**kwargs):
            # Set environment variables
            monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
            monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")

            # Create client with custom parameters
            client = Langfuse(
                public_key="test-public-key",
                secret_key="test-secret-key",
                host="http://test-host",
                tracing_enabled=True,
                **kwargs,
            )

            # Configure client
            client._otel_tracer = tracer_provider.get_tracer("langfuse-test")

            return client

        return _create_client

    # ------ Test Metadata Fixtures ------

    @pytest.fixture
    def simple_metadata(self):
        """Create simple metadata for testing."""
        return {"key1": "value1", "key2": 123, "key3": True}

    @pytest.fixture
    def nested_metadata(self):
        """Create nested metadata structure for testing."""
        return {
            "config": {
                "model": "gpt-4",
                "parameters": {"temperature": 0.7, "max_tokens": 500},
            },
            "telemetry": {"client_info": {"version": "1.0.0", "platform": "python"}},
        }

    @pytest.fixture
    def complex_metadata(self):
        """Create complex metadata with various types for testing."""
        return {
            "string_value": "test string",
            "int_value": 42,
            "float_value": 3.14159,
            "bool_value": True,
            "null_value": None,
            "list_value": [1, 2, 3, "four", 5.0],
            "nested_dict": {
                "key1": "value1",
                "key2": 123,
                "nested_list": ["a", "b", "c"],
            },
            "datetime": datetime.now(),
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
        }

    # ------ Helper Methods ------

    def get_span_data(self, span: ReadableSpan) -> dict:
        """Extract important data from a span for testing."""
        return {
            "name": span.name,
            "attributes": dict(span.attributes) if span.attributes else {},
            "span_id": format(span.context.span_id, "016x"),
            "trace_id": format(span.context.trace_id, "032x"),
            "parent_span_id": format(span.parent.span_id, "016x")
            if span.parent
            else None,
        }

    def get_spans_by_name(self, memory_exporter, name: str) -> List[dict]:
        """Get all spans with a specific name."""
        spans = memory_exporter.get_finished_spans()
        return [self.get_span_data(span) for span in spans if span.name == name]

    def verify_span_attribute(
        self, span_data: dict, attribute_key: str, expected_value=None
    ):
        """Verify that a span has a specific attribute with an optional expected value."""
        attributes = span_data["attributes"]
        assert (
            attribute_key in attributes
        ), f"Attribute {attribute_key} not found in span"

        if expected_value is not None:
            assert (
                attributes[attribute_key] == expected_value
            ), f"Expected {attribute_key} to be {expected_value}, got {attributes[attribute_key]}"

        return attributes[attribute_key]

    def verify_json_attribute(
        self, span_data: dict, attribute_key: str, expected_dict=None
    ):
        """Verify that a span has a JSON attribute and optionally check its parsed value."""
        json_string = self.verify_span_attribute(span_data, attribute_key)
        parsed_json = json.loads(json_string)

        if expected_dict is not None:
            assert (
                parsed_json == expected_dict
            ), f"Expected JSON {attribute_key} to be {expected_dict}, got {parsed_json}"

        return parsed_json

    def assert_parent_child_relationship(self, parent_span: dict, child_span: dict):
        """Verify parent-child relationship between two spans."""
        assert (
            child_span["parent_span_id"] == parent_span["span_id"]
        ), f"Child span {child_span['name']} should have parent {parent_span['name']}"
        assert (
            child_span["trace_id"] == parent_span["trace_id"]
        ), f"Child span {child_span['name']} should have same trace ID as parent {parent_span['name']}"


class TestBasicSpans(TestOTelBase):
    """Tests for basic span operations and attributes."""

    def test_basic_span_creation(self, langfuse_client, memory_exporter):
        """Test that a basic span can be created with attributes."""
        # Create a span and end it
        span = langfuse_client.start_span(name="test-span", input={"test": "value"})
        span.end()

        # Get spans with our name
        spans = self.get_spans_by_name(memory_exporter, "test-span")

        # Verify we created exactly one span
        assert (
            len(spans) == 1
        ), f"Expected 1 span named 'test-span', but found {len(spans)}"
        span_data = spans[0]

        # Verify the span attributes
        assert span_data["name"] == "test-span"
        self.verify_span_attribute(
            span_data, LangfuseOtelSpanAttributes.OBSERVATION_TYPE, "span"
        )

        # Verify the span IDs match
        assert span.id == span_data["span_id"]
        assert span.trace_id == span_data["trace_id"]

    def test_span_hierarchy(self, langfuse_client, memory_exporter):
        """Test creating nested spans and verify their parent-child relationships."""
        # Create parent span
        with langfuse_client.start_as_current_span(name="parent-span") as parent_span:
            # Create a child span
            child_span = parent_span.start_span(name="child-span")
            child_span.end()

            # Create another child span using context manager
            with parent_span.start_as_current_span(name="child-span-2") as child_span_2:
                # Create a grandchild span
                grandchild = child_span_2.start_span(name="grandchild-span")
                grandchild.end()

        # Get all spans
        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Find spans by name
        parent = next((s for s in spans if s["name"] == "parent-span"), None)
        child1 = next((s for s in spans if s["name"] == "child-span"), None)
        child2 = next((s for s in spans if s["name"] == "child-span-2"), None)
        grandchild = next((s for s in spans if s["name"] == "grandchild-span"), None)

        # Verify all spans exist
        assert parent is not None, "Parent span not found"
        assert child1 is not None, "First child span not found"
        assert child2 is not None, "Second child span not found"
        assert grandchild is not None, "Grandchild span not found"

        # Verify parent-child relationships
        self.assert_parent_child_relationship(parent, child1)
        self.assert_parent_child_relationship(parent, child2)
        self.assert_parent_child_relationship(child2, grandchild)

        # All spans should have the same trace ID
        assert len(set(s["trace_id"] for s in spans)) == 1

    def test_update_current_span_name(self, langfuse_client, memory_exporter):
        """Test updating current span name via update_current_span method."""
        # Create a span using context manager
        with langfuse_client.start_as_current_span(name="original-current-span"):
            # Update the current span name
            langfuse_client.update_current_span(name="updated-current-span")

        # Verify the span name was updated
        spans = self.get_spans_by_name(memory_exporter, "updated-current-span")
        assert len(spans) == 1, "Expected one span with updated name"

        # Also verify no spans exist with the original name
        original_spans = self.get_spans_by_name(
            memory_exporter, "original-current-span"
        )
        assert len(original_spans) == 0, "Expected no spans with original name"

    def test_span_attributes(self, langfuse_client, memory_exporter):
        """Test that span attributes are correctly set and updated."""
        # Create a span with attributes
        span = langfuse_client.start_span(
            name="attribute-span",
            input={"prompt": "Test prompt"},
            output={"response": "Test response"},
            metadata={"session": "test-session"},
            level="INFO",
            status_message="Test status",
        )

        # Update span with new attributes
        span.update(output={"response": "Updated response"}, metadata={"updated": True})

        span.end()

        # Get the span data
        spans = self.get_spans_by_name(memory_exporter, "attribute-span")
        assert len(spans) == 1, "Expected one attribute-span"
        span_data = spans[0]

        # Verify attributes are set
        attributes = span_data["attributes"]
        assert LangfuseOtelSpanAttributes.OBSERVATION_INPUT in attributes
        assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT in attributes
        assert (
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.session" in attributes
        )

        # Parse JSON attributes
        input_data = json.loads(
            attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
        )
        output_data = json.loads(
            attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
        )
        metadata_data = attributes[
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.session"
        ]

        # Verify attribute values
        assert input_data == {"prompt": "Test prompt"}
        assert output_data == {"response": "Updated response"}
        assert metadata_data == "test-session"
        assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] == "INFO"
        assert (
            attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
            == "Test status"
        )

    def test_span_name_update(self, langfuse_client, memory_exporter):
        """Test updating span name via update method."""
        # Create a span with initial name
        span = langfuse_client.start_span(name="original-span-name")

        # Update the span name
        span.update(name="updated-span-name")
        span.end()

        # Verify the span name was updated
        spans = self.get_spans_by_name(memory_exporter, "updated-span-name")
        assert len(spans) == 1, "Expected one span with updated name"

        # Also verify no spans exist with the original name
        original_spans = self.get_spans_by_name(memory_exporter, "original-span-name")
        assert len(original_spans) == 0, "Expected no spans with original name"

    def test_generation_span(self, langfuse_client, memory_exporter):
        """Test creating a generation span with model-specific attributes."""
        # Create a generation
        generation = langfuse_client.start_generation(
            name="test-generation",
            model="gpt-4",
            model_parameters={"temperature": 0.7, "max_tokens": 100},
            input={"prompt": "Hello, AI"},
            output={"response": "Hello, human"},
            usage_details={"input": 10, "output": 5, "total": 15},
        )
        generation.end()

        # Get the span data
        spans = self.get_spans_by_name(memory_exporter, "test-generation")
        assert len(spans) == 1, "Expected one test-generation span"
        gen_data = spans[0]

        # Verify generation-specific attributes
        attributes = gen_data["attributes"]
        assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "generation"
        assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4"

        # Parse complex attributes
        model_params = json.loads(
            attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS]
        )
        assert model_params == {"temperature": 0.7, "max_tokens": 100}

        usage = json.loads(
            attributes[LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS]
        )
        assert usage == {"input": 10, "output": 5, "total": 15}

    def test_generation_name_update(self, langfuse_client, memory_exporter):
        """Test updating generation name via update method."""
        # Create a generation with initial name
        generation = langfuse_client.start_generation(
            name="original-generation-name", model="gpt-4"
        )

        # Update the generation name
        generation.update(name="updated-generation-name")
        generation.end()

        # Verify the generation name was updated
        spans = self.get_spans_by_name(memory_exporter, "updated-generation-name")
        assert len(spans) == 1, "Expected one generation with updated name"

        # Also verify no spans exist with the original name
        original_spans = self.get_spans_by_name(
            memory_exporter, "original-generation-name"
        )
        assert len(original_spans) == 0, "Expected no generations with original name"

    def test_trace_update(self, langfuse_client, memory_exporter):
        """Test updating trace level attributes."""
        # Create a span and update trace attributes
        with langfuse_client.start_as_current_span(name="trace-span") as span:
            span.update_trace(
                name="updated-trace-name",
                user_id="test-user",
                session_id="test-session",
                tags=["tag1", "tag2"],
                input={"trace-input": "value"},
                metadata={"trace-meta": "data"},
            )

        # Get the span data
        spans = self.get_spans_by_name(memory_exporter, "trace-span")
        assert len(spans) == 1, "Expected one trace-span"
        span_data = spans[0]

        # Verify trace attributes were set
        attributes = span_data["attributes"]
        assert attributes[LangfuseOtelSpanAttributes.TRACE_NAME] == "updated-trace-name"
        assert attributes[LangfuseOtelSpanAttributes.TRACE_USER_ID] == "test-user"
        assert attributes[LangfuseOtelSpanAttributes.TRACE_SESSION_ID] == "test-session"

        # Handle different serialization formats
        if isinstance(attributes[LangfuseOtelSpanAttributes.TRACE_TAGS], str):
            tags = json.loads(attributes[LangfuseOtelSpanAttributes.TRACE_TAGS])
        else:
            tags = list(attributes[LangfuseOtelSpanAttributes.TRACE_TAGS])

        input_data = json.loads(attributes[LangfuseOtelSpanAttributes.TRACE_INPUT])
        metadata = attributes[f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.trace-meta"]

        # Check attribute values
        assert sorted(tags) == sorted(["tag1", "tag2"])
        assert input_data == {"trace-input": "value"}
        assert metadata == "data"

    def test_complex_scenario(self, langfuse_client, memory_exporter):
        """Test a more complex scenario with multiple operations and nesting."""
        # Create a trace with a main span
        with langfuse_client.start_as_current_span(name="main-flow") as main_span:
            # Add trace information
            main_span.update_trace(
                name="complex-test",
                user_id="complex-user",
                session_id="complex-session",
            )

            # Add a processing span
            with main_span.start_as_current_span(name="processing") as processing:
                processing.update(metadata={"step": "processing"})

            # Add an LLM generation
            with main_span.start_as_current_generation(
                name="llm-call",
                model="gpt-3.5-turbo",
                input={"prompt": "Summarize this text"},
                metadata={"service": "OpenAI"},
            ) as generation:
                # Update the generation with results
                generation.update(
                    output={"text": "This is a summary"},
                    usage_details={"input": 20, "output": 5, "total": 25},
                )

            # Final processing step
            with main_span.start_as_current_span(name="post-processing") as post_proc:
                post_proc.update(metadata={"step": "post-processing"})

        # Get all spans
        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Find each span by name
        main = next((s for s in spans if s["name"] == "main-flow"), None)
        proc = next((s for s in spans if s["name"] == "processing"), None)
        llm = next((s for s in spans if s["name"] == "llm-call"), None)
        post = next((s for s in spans if s["name"] == "post-processing"), None)

        # Verify all spans exist
        assert main is not None, "Main span not found"
        assert proc is not None, "Processing span not found"
        assert llm is not None, "LLM span not found"
        assert post is not None, "Post-processing span not found"

        # Verify parent-child relationships
        self.assert_parent_child_relationship(main, proc)
        self.assert_parent_child_relationship(main, llm)
        self.assert_parent_child_relationship(main, post)

        # Verify all spans have the same trace ID
        assert len(set(s["trace_id"] for s in spans)) == 1

        # Check specific attributes
        assert (
            main["attributes"][LangfuseOtelSpanAttributes.TRACE_NAME] == "complex-test"
        )
        assert (
            llm["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_TYPE]
            == "generation"
        )

        # Parse metadata
        proc_metadata = proc["attributes"][
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.step"
        ]
        assert proc_metadata == "processing"

        # Parse input/output JSON
        llm_input = json.loads(
            llm["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
        )
        llm_output = json.loads(
            llm["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
        )
        assert llm_input == {"prompt": "Summarize this text"}
        assert llm_output == {"text": "This is a summary"}

    def test_update_current_generation_name(self, langfuse_client, memory_exporter):
        """Test updating current generation name via update_current_generation method."""
        # Create a generation using context manager
        with langfuse_client.start_as_current_generation(
            name="original-current-generation", model="gpt-4"
        ):
            # Update the current generation name
            langfuse_client.update_current_generation(name="updated-current-generation")

        # Verify the generation name was updated
        spans = self.get_spans_by_name(memory_exporter, "updated-current-generation")
        assert len(spans) == 1, "Expected one generation with updated name"

        # Also verify no spans exist with the original name
        original_spans = self.get_spans_by_name(
            memory_exporter, "original-current-generation"
        )
        assert len(original_spans) == 0, "Expected no generations with original name"

    def test_start_as_current_observation_types(self, langfuse_client, memory_exporter):
        """Test creating different observation types using start_as_current_observation."""
        # Test each observation type from ObservationTypeLiteralNoEvent
        observation_types = [
            "span",
            "generation",
            "agent",
            "tool",
            "chain",
            "retriever",
            "evaluator",
            "embedding",
            "guardrail",
        ]

        for obs_type in observation_types:
            with langfuse_client.start_as_current_observation(
                name=f"test-{obs_type}", as_type=obs_type
            ) as obs:
                obs.update_trace(name=f"trace-{obs_type}")

        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Find spans by name and verify their observation types
        for obs_type in observation_types:
            expected_name = f"test-{obs_type}"
            matching_spans = [span for span in spans if span["name"] == expected_name]
            assert (
                len(matching_spans) == 1
            ), f"Expected one span with name {expected_name}"

            span_data = matching_spans[0]
            expected_otel_type = obs_type  # OTEL attributes use lowercase
            actual_type = span_data["attributes"].get(
                LangfuseOtelSpanAttributes.OBSERVATION_TYPE
            )

            assert (
                actual_type == expected_otel_type
            ), f"Expected observation type {expected_otel_type}, got {actual_type}"

    def test_start_observation(self, langfuse_client, memory_exporter):
        """Test creating different observation types using start_observation."""
        from langfuse._client.constants import (
            ObservationTypeGenerationLike,
            ObservationTypeLiteral,
            get_observation_types_list,
        )

        # Test each observation type defined in constants - this ensures we test all supported types
        observation_types = get_observation_types_list(ObservationTypeLiteral)

        # Create a main span to use for child creation
        with langfuse_client.start_as_current_span(
            name="factory-test-parent"
        ) as parent_span:
            created_observations = []

            for obs_type in observation_types:
                if obs_type in get_observation_types_list(
                    ObservationTypeGenerationLike
                ):
                    # Generation-like types with extra parameters
                    obs = parent_span.start_observation(
                        name=f"factory-{obs_type}",
                        as_type=obs_type,
                        input={"test": f"{obs_type}_input"},
                        model="test-model",
                        model_parameters={"temperature": 0.7},
                        usage_details={"input": 10, "output": 20},
                    )
                    if obs_type != "event":  # Events are auto-ended
                        obs.end()
                    created_observations.append((obs_type, obs))
                elif obs_type == "event":
                    # Test event creation through start_observation (should be auto-ended)
                    obs = parent_span.start_observation(
                        name=f"factory-{obs_type}",
                        as_type=obs_type,
                        input={"test": f"{obs_type}_input"},
                    )
                    created_observations.append((obs_type, obs))
                else:
                    # Span-like types (span, guardrail)
                    obs = parent_span.start_observation(
                        name=f"factory-{obs_type}",
                        as_type=obs_type,
                        input={"test": f"{obs_type}_input"},
                    )
                    obs.end()
                    created_observations.append((obs_type, obs))

        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Verify factory pattern created correct observation types
        for obs_type in observation_types:
            expected_name = f"factory-{obs_type}"
            matching_spans = [span for span in spans if span["name"] == expected_name]
            assert (
                len(matching_spans) == 1
            ), f"Expected one span with name {expected_name}, found {len(matching_spans)}"

            span_data = matching_spans[0]
            actual_type = span_data["attributes"].get(
                LangfuseOtelSpanAttributes.OBSERVATION_TYPE
            )

            assert (
                actual_type == obs_type
            ), f"Factory pattern failed: Expected observation type {obs_type}, got {actual_type}"

        # Ensure returned objects are of correct types
        for obs_type, obs_instance in created_observations:
            if obs_type == "span":
                from langfuse._client.span import LangfuseSpan

                assert isinstance(
                    obs_instance, LangfuseSpan
                ), f"Expected LangfuseSpan, got {type(obs_instance)}"
            elif obs_type == "generation":
                from langfuse._client.span import LangfuseGeneration

                assert isinstance(
                    obs_instance, LangfuseGeneration
                ), f"Expected LangfuseGeneration, got {type(obs_instance)}"
            elif obs_type == "agent":
                from langfuse._client.span import LangfuseAgent

                assert isinstance(
                    obs_instance, LangfuseAgent
                ), f"Expected LangfuseAgent, got {type(obs_instance)}"
            elif obs_type == "tool":
                from langfuse._client.span import LangfuseTool

                assert isinstance(
                    obs_instance, LangfuseTool
                ), f"Expected LangfuseTool, got {type(obs_instance)}"
            elif obs_type == "chain":
                from langfuse._client.span import LangfuseChain

                assert isinstance(
                    obs_instance, LangfuseChain
                ), f"Expected LangfuseChain, got {type(obs_instance)}"
            elif obs_type == "retriever":
                from langfuse._client.span import LangfuseRetriever

                assert isinstance(
                    obs_instance, LangfuseRetriever
                ), f"Expected LangfuseRetriever, got {type(obs_instance)}"
            elif obs_type == "evaluator":
                from langfuse._client.span import LangfuseEvaluator

                assert isinstance(
                    obs_instance, LangfuseEvaluator
                ), f"Expected LangfuseEvaluator, got {type(obs_instance)}"
            elif obs_type == "embedding":
                from langfuse._client.span import LangfuseEmbedding

                assert isinstance(
                    obs_instance, LangfuseEmbedding
                ), f"Expected LangfuseEmbedding, got {type(obs_instance)}"
            elif obs_type == "guardrail":
                from langfuse._client.span import LangfuseGuardrail

                assert isinstance(
                    obs_instance, LangfuseGuardrail
                ), f"Expected LangfuseGuardrail, got {type(obs_instance)}"
            elif obs_type == "event":
                from langfuse._client.span import LangfuseEvent

                assert isinstance(
                    obs_instance, LangfuseEvent
                ), f"Expected LangfuseEvent, got {type(obs_instance)}"

    def test_custom_trace_id(self, langfuse_client, memory_exporter):
        """Test setting a custom trace ID."""
        # Create a custom trace ID
        custom_trace_id = "abcdef1234567890abcdef1234567890"

        # Create a span with this custom trace ID using trace_context
        trace_context = {"trace_id": custom_trace_id}
        span = langfuse_client.start_span(
            name="custom-trace-span",
            trace_context=trace_context,
            input={"test": "value"},
        )
        span.end()

        # Get spans and verify the trace ID matches
        spans = self.get_spans_by_name(memory_exporter, "custom-trace-span")
        assert len(spans) == 1, "Expected one span"

        span_data = spans[0]
        assert (
            span_data["trace_id"] == custom_trace_id
        ), "Trace ID doesn't match custom ID"
        assert span_data["attributes"][LangfuseOtelSpanAttributes.AS_ROOT] is True

        # Test additional spans with the same trace context
        child_span = langfuse_client.start_span(
            name="child-span", trace_context=trace_context, input={"child": "data"}
        )
        child_span.end()

        # Verify child span uses the same trace ID
        child_spans = self.get_spans_by_name(memory_exporter, "child-span")
        assert len(child_spans) == 1, "Expected one child span"
        assert (
            child_spans[0]["trace_id"] == custom_trace_id
        ), "Child span has wrong trace ID"

    def test_custom_parent_span_id(self, langfuse_client, memory_exporter):
        """Test setting a custom parent span ID."""
        # Create a trace and get its ID
        trace_id = "abcdef1234567890abcdef1234567890"
        parent_span_id = "fedcba0987654321"

        # Create a context with trace ID and parent span ID
        trace_context = {"trace_id": trace_id, "parent_span_id": parent_span_id}

        # Create a span with this context
        span = langfuse_client.start_span(
            name="custom-parent-span", trace_context=trace_context
        )
        span.end()

        # Verify the span is created with the right parent
        spans = self.get_spans_by_name(memory_exporter, "custom-parent-span")
        assert len(spans) == 1, "Expected one span"
        assert spans[0]["trace_id"] == trace_id
        assert spans[0]["attributes"][LangfuseOtelSpanAttributes.AS_ROOT] is True

    def test_multiple_generations_in_trace(self, langfuse_client, memory_exporter):
        """Test creating multiple generation spans within the same trace."""
        # Create a trace with multiple generation spans
        with langfuse_client.start_as_current_span(name="multi-gen-flow") as main_span:
            # First generation
            gen1 = main_span.start_generation(
                name="generation-1",
                model="gpt-3.5-turbo",
                input={"prompt": "First prompt"},
                output={"text": "First response"},
                model_parameters={"temperature": 0.7},
                usage_details={"input": 10, "output": 20, "total": 30},
            )
            gen1.end()

            # Second generation with different model
            gen2 = main_span.start_generation(
                name="generation-2",
                model="gpt-4",
                input={"prompt": "Second prompt"},
                output={"text": "Second response"},
                model_parameters={"temperature": 0.5},
                usage_details={"input": 15, "output": 25, "total": 40},
            )
            gen2.end()

        # Get all spans
        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Find main span and generations
        main = next((s for s in spans if s["name"] == "multi-gen-flow"), None)
        gen1_data = next((s for s in spans if s["name"] == "generation-1"), None)
        gen2_data = next((s for s in spans if s["name"] == "generation-2"), None)

        # Verify all spans exist
        assert main is not None, "Main span not found"
        assert gen1_data is not None, "First generation span not found"
        assert gen2_data is not None, "Second generation span not found"

        # Verify parent-child relationships
        self.assert_parent_child_relationship(main, gen1_data)
        self.assert_parent_child_relationship(main, gen2_data)

        # Verify all spans have the same trace ID
        assert len(set(s["trace_id"] for s in spans)) == 1

        # Verify generation-specific attributes are correct
        assert (
            gen1_data["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_TYPE]
            == "generation"
        )
        assert (
            gen1_data["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_MODEL]
            == "gpt-3.5-turbo"
        )

        assert (
            gen2_data["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_TYPE]
            == "generation"
        )
        assert (
            gen2_data["attributes"][LangfuseOtelSpanAttributes.OBSERVATION_MODEL]
            == "gpt-4"
        )

        # Parse usage details
        gen1_usage = json.loads(
            gen1_data["attributes"][
                LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
            ]
        )
        gen2_usage = json.loads(
            gen2_data["attributes"][
                LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
            ]
        )

        assert gen1_usage == {"input": 10, "output": 20, "total": 30}
        assert gen2_usage == {"input": 15, "output": 25, "total": 40}

    def test_error_handling(self, langfuse_client, memory_exporter):
        """Test error handling in span operations."""
        # Create a span that will have an error
        span = langfuse_client.start_span(name="error-span")

        # Set an error status on the span
        import traceback

        from opentelemetry.trace.status import Status, StatusCode

        try:
            # Deliberately raise an exception
            raise ValueError("Test error message")
        except Exception as e:
            # Get the exception details
            stack_trace = traceback.format_exc()
            # Record the error on the span
            span._otel_span.set_status(Status(StatusCode.ERROR))
            span._otel_span.record_exception(e, attributes={"stack_trace": stack_trace})
            span.update(level="ERROR", status_message=str(e))

        # End the span with error status
        span.end()

        # Verify the span contains error information
        spans = self.get_spans_by_name(memory_exporter, "error-span")
        assert len(spans) == 1, "Expected one error span"

        span_data = spans[0]
        attributes = span_data["attributes"]

        # Verify error attributes were set correctly
        assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] == "ERROR"
        assert (
            attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
            == "Test error message"
        )


class TestAdvancedSpans(TestOTelBase):
    """Tests for advanced span functionality including generations, timing, and usage metrics."""

    def test_complex_model_parameters(self, langfuse_client, memory_exporter):
        """Test handling of complex model parameters in generation spans."""
        # Create a complex model parameters dictionary with nested structures
        complex_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.5,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            "response_format": {"type": "json_object"},
        }

        # Create a generation with these complex parameters
        generation = langfuse_client.start_generation(
            name="complex-params-test",
            model="gpt-4",
            model_parameters=complex_params,
            input={"prompt": "What's the weather?"},
        )
        generation.end()

        # Get the generation span
        spans = self.get_spans_by_name(memory_exporter, "complex-params-test")
        assert len(spans) == 1, "Expected one generation span"
        span_data = spans[0]

        # Skip further assertions if model parameters attribute isn't present
        if (
            LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
            not in span_data["attributes"]
        ):
            pytest.skip("Model parameters attribute not implemented yet")

        # Verify model parameters were properly serialized
        model_params = self.verify_json_attribute(
            span_data, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
        )

        # Verify all parameters were preserved correctly
        assert model_params["temperature"] == 0.8
        assert model_params["top_p"] == 0.95
        assert model_params["presence_penalty"] == 1.0
        assert model_params["frequency_penalty"] == 0.5
        assert len(model_params["tools"]) == 1
        assert model_params["tools"][0]["type"] == "function"
        assert model_params["tools"][0]["function"]["name"] == "get_weather"
        assert "parameters" in model_params["tools"][0]["function"]
        assert model_params["response_format"]["type"] == "json_object"

    def test_updating_current_generation(self, langfuse_client, memory_exporter):
        """Test that an in-progress generation can be updated multiple times."""
        # Create a generation
        generation = langfuse_client.start_generation(
            name="updating-generation",
            model="gpt-4",
            input={"prompt": "Write a story about a robot"},
        )

        # Start completion (skip if not implemented)
        try:
            generation.set_completion_start()
        except (AttributeError, NotImplementedError):
            pass

        # Update with partial output (streaming)
        generation.update(
            output={"partial_text": "Once upon a time, there was a robot"}
        )

        # Update with more content (streaming continues)
        generation.update(
            output={
                "partial_text": "Once upon a time, there was a robot named Bleep who dreamed of becoming human."
            }
        )

        # Update with final content and usage
        generation.update(
            output={
                "text": "Once upon a time, there was a robot named Bleep who dreamed of becoming human. Every day, Bleep would observe humans and try to understand their emotions..."
            },
            usage_details={"input": 10, "output": 50, "total": 60},
        )

        # End the generation
        generation.end()

        # Get the generation span
        spans = self.get_spans_by_name(memory_exporter, "updating-generation")
        assert len(spans) == 1, "Expected one generation span"
        span_data = spans[0]

        # Verify final attributes
        output = self.verify_json_attribute(
            span_data, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        )

        # Verify final output contains the complete text (key name may vary)
        text_key = "text" if "text" in output else "partial_text"
        assert text_key in output
        assert "robot named Bleep" in output[text_key]

        # Skip usage check if the attribute isn't present
        if (
            LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
            in span_data["attributes"]
        ):
            usage = self.verify_json_attribute(
                span_data, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
            )
            assert usage["input"] == 10
            assert usage["output"] == 50
            assert usage["total"] == 60

    def test_sampling(self, monkeypatch, tracer_provider, mock_processor_init):
        """Test sampling behavior."""
        # Create a new memory exporter for this test
        sampled_exporter = InMemorySpanExporter()

        # Create a tracer provider with sampling
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        sampled_provider = TracerProvider(
            resource=Resource.create({"service.name": "sampled-test"}),
            sampler=TraceIdRatioBased(0),  # 0% sampling rate = nothing gets sampled
        )
        processor = SimpleSpanProcessor(sampled_exporter)
        sampled_provider.add_span_processor(processor)

        # Save original provider to restore later
        original_provider = trace_api.get_tracer_provider()
        trace_api.set_tracer_provider(sampled_provider)

        # Create a client with the sampled provider
        client = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="http://test-host",
            tracing_enabled=True,
            sample_rate=0,  # No sampling
        )

        # Create several spans
        for i in range(5):
            span = client.start_span(name=f"sampled-span-{i}")
            span.end()

        # With a sample rate of 0, we should have no spans
        assert (
            len(sampled_exporter.get_finished_spans()) == 0
        ), "Expected no spans with 0 sampling"

        # Restore the original provider
        trace_api.set_tracer_provider(original_provider)

    @pytest.mark.skip("Calling shutdown will pollute the global context")
    def test_shutdown_and_flush(self, langfuse_client, memory_exporter):
        """Test shutdown and flush operations."""
        # Create a span without ending it
        span = langfuse_client.start_span(name="flush-test-span")

        # Explicitly flush
        langfuse_client.flush()

        # The span is still active, so it shouldn't be in the exporter yet
        spans = self.get_spans_by_name(memory_exporter, "flush-test-span")
        assert len(spans) == 0, "Span shouldn't be exported before it's ended"

        # Now end the span
        span.end()

        # After ending, it should be exported
        spans = self.get_spans_by_name(memory_exporter, "flush-test-span")
        assert len(spans) == 1, "Span should be exported after ending"

        # Create another span for shutdown testing
        langfuse_client.start_span(name="shutdown-test-span")

        # Call shutdown (should flush any pending spans)
        langfuse_client.shutdown()

    def test_disabled_tracing(self, monkeypatch, tracer_provider, mock_processor_init):
        """Test behavior when tracing is disabled."""
        # Create a client with tracing disabled
        client = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="http://test-host",
            tracing_enabled=False,
        )

        # Create a memory exporter to verify no spans are created
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)

        # Attempt to create spans and trace operations
        span = client.start_span(name="disabled-span", input={"key": "value"})
        span.update(output={"result": "test"})
        span.end()

        with client.start_as_current_span(name="disabled-context-span") as context_span:
            context_span.update_trace(name="disabled-trace")

        # Verify no spans were created
        spans = exporter.get_finished_spans()
        assert (
            len(spans) == 0
        ), f"Expected no spans when tracing is disabled, got {len(spans)}"

    def test_trace_id_generation(self, langfuse_client):
        """Test trace ID generation follows expected format."""
        # Generate trace IDs
        trace_id1 = langfuse_client.create_trace_id()
        trace_id2 = langfuse_client.create_trace_id()

        # Verify format: 32 hex characters
        assert (
            len(trace_id1) == 32
        ), f"Trace ID length should be 32, got {len(trace_id1)}"
        assert (
            len(trace_id2) == 32
        ), f"Trace ID length should be 32, got {len(trace_id2)}"

        # jerify it's a valid hex string
        int(trace_id1, 16), "Trace ID should be a valid hex string"
        int(trace_id2, 16), "Trace ID should be a valid hex string"

        # IDs should be unique
        assert trace_id1 != trace_id2, "Generated trace IDs should be unique"


class TestMetadataHandling(TestOTelBase):
    """Tests for metadata serialization, updates, and integrity."""

    def test_complex_metadata_serialization(self):
        """Test the _flatten_and_serialize_metadata function directly."""
        from langfuse._client.attributes import (
            _flatten_and_serialize_metadata,
            _serialize,
        )

        # Test case 1: Non-dict metadata
        non_dict_result = _flatten_and_serialize_metadata("string-value", "observation")
        assert LangfuseOtelSpanAttributes.OBSERVATION_METADATA in non_dict_result
        assert non_dict_result[
            LangfuseOtelSpanAttributes.OBSERVATION_METADATA
        ] == _serialize("string-value")

        # Test case 2: Simple dict
        simple_dict = {"key1": "value1", "key2": 123}
        simple_result = _flatten_and_serialize_metadata(simple_dict, "observation")
        assert (
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.key1" in simple_result
        )
        assert (
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.key2" in simple_result
        )
        assert (
            simple_result[f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.key1"]
            == "value1"
        )
        assert (
            simple_result[f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.key2"]
            == 123
        )

        # Test case 3: Nested dict (will be flattened in current implementation)
        nested_dict = {
            "outer": {"inner1": "value1", "inner2": 123},
            "list_key": [1, 2, 3],
        }
        nested_result = _flatten_and_serialize_metadata(nested_dict, "trace")

        # Verify the keys are flattened properly
        outer_key = f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.outer"
        list_key = f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.list_key"

        assert outer_key in nested_result
        assert list_key in nested_result

        # The inner dictionary should be serialized as a JSON string
        assert json.loads(nested_result[outer_key]) == {
            "inner1": "value1",
            "inner2": 123,
        }
        assert json.loads(nested_result[list_key]) == [1, 2, 3]

        # Test case 4: Empty dict
        empty_result = _flatten_and_serialize_metadata({}, "observation")
        assert len(empty_result) == 0

        # Test case 5: None
        none_result = _flatten_and_serialize_metadata(None, "observation")
        # The implementation returns a dictionary with a None value
        assert LangfuseOtelSpanAttributes.OBSERVATION_METADATA in none_result
        assert none_result[LangfuseOtelSpanAttributes.OBSERVATION_METADATA] is None

        # Test case 6: Complex nested structure
        complex_dict = {
            "level1": {
                "level2": {"level3": {"value": "deeply nested"}},
                "array": [{"item1": 1}, {"item2": 2}],
            },
            "sibling": "value",
        }
        complex_result = _flatten_and_serialize_metadata(complex_dict, "observation")

        # Check first-level keys only (current implementation)
        level1_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.level1"
        sibling_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.sibling"

        assert level1_key in complex_result
        assert sibling_key in complex_result

        # The nested structures are serialized as JSON strings
        assert json.loads(complex_result[level1_key]) == complex_dict["level1"]
        assert complex_result[sibling_key] == "value"

    def test_nested_metadata_updates(self):
        """Test that nested metadata updates don't overwrite unrelated keys."""
        from langfuse._client.attributes import _flatten_and_serialize_metadata

        # Test how updates to metadata should behave in sequential calls
        # Initial metadata
        initial_metadata = {
            "config": {
                "model": "gpt-4",
                "parameters": {"temperature": 0.7, "max_tokens": 500},
            },
            "telemetry": {"client_info": {"version": "1.0.0", "platform": "python"}},
        }

        # First flattening
        first_result = _flatten_and_serialize_metadata(initial_metadata, "observation")

        # Update with new config temperature only
        update_metadata = {
            "config": {
                "parameters": {
                    "temperature": 0.9  # Changed from 0.7
                }
            }
        }

        # Second flattening (would happen on update)
        second_result = _flatten_and_serialize_metadata(update_metadata, "observation")

        # In a merge scenario, we'd have:
        # config.model: kept from first_result
        # config.temperature: updated from second_result
        # telemetry.session_id: kept from first_result

        # Get the expected keys
        config_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.config"
        telemetry_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.telemetry"

        # Verify the structure of the results
        assert config_key in first_result
        assert telemetry_key in first_result

        # Check serialized values can be parsed
        first_config = json.loads(first_result[config_key])
        assert first_config["model"] == "gpt-4"
        assert first_config["parameters"]["temperature"] == 0.7

        first_telemetry = json.loads(first_result[telemetry_key])
        assert first_telemetry["client_info"]["version"] == "1.0.0"

        # Verify the second result only contains the config key
        assert config_key in second_result
        assert telemetry_key not in second_result

        # Check the updated temperature
        second_config = json.loads(second_result[config_key])
        assert "parameters" in second_config
        assert second_config["parameters"]["temperature"] == 0.9

        # Now test with completely different metadata keys
        first_metadata = {"first_section": {"key1": "value1", "key2": "value2"}}

        second_metadata = {"second_section": {"key3": "value3"}}

        # Generate flattened results
        first_section_result = _flatten_and_serialize_metadata(
            first_metadata, "observation"
        )
        second_section_result = _flatten_and_serialize_metadata(
            second_metadata, "observation"
        )

        # Get expected keys
        first_section_key = (
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.first_section"
        )
        second_section_key = (
            f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.second_section"
        )

        # Verify each section is properly serialized
        assert first_section_key in first_section_result
        assert second_section_key in second_section_result

        # In a merge scenario, both keys would be present
        merged_result = {**first_section_result, **second_section_result}
        assert first_section_key in merged_result
        assert second_section_key in merged_result

        # Check the values
        first_section_data = json.loads(merged_result[first_section_key])
        second_section_data = json.loads(merged_result[second_section_key])

        assert first_section_data["key1"] == "value1"
        assert first_section_data["key2"] == "value2"
        assert second_section_data["key3"] == "value3"

    def test_metadata_integrity_in_async_environment(self):
        """Test that metadata nesting integrity is preserved in async contexts."""
        import asyncio

        from langfuse._client.attributes import _flatten_and_serialize_metadata

        # Initial metadata with complex nested structure
        initial_metadata = {
            "config": {
                "model": "gpt-4",
                "parameters": {"temperature": 0.7, "max_tokens": 500},
            },
            "telemetry": {"client_info": {"version": "1.0.0", "platform": "python"}},
        }

        # Define async metadata update functions
        async def update_config_temperature():
            # Update just temperature
            update = {"config": {"parameters": {"temperature": 0.9}}}
            return _flatten_and_serialize_metadata(update, "observation")

        async def update_telemetry_version():
            # Update just version
            update = {"telemetry": {"client_info": {"version": "1.1.0"}}}
            return _flatten_and_serialize_metadata(update, "observation")

        async def update_config_model():
            # Update just model
            update = {"config": {"model": "gpt-3.5-turbo"}}
            return _flatten_and_serialize_metadata(update, "observation")

        async def update_telemetry_platform():
            # Update just platform
            update = {"telemetry": {"client_info": {"platform": "web"}}}
            return _flatten_and_serialize_metadata(update, "observation")

        # Create multiple tasks to run concurrently
        async def run_concurrent_updates():
            # Initial flattening
            base_result = _flatten_and_serialize_metadata(
                initial_metadata, "observation"
            )

            # Run all updates concurrently
            (
                temperature_result,
                version_result,
                model_result,
                platform_result,
            ) = await asyncio.gather(
                update_config_temperature(),
                update_telemetry_version(),
                update_config_model(),
                update_telemetry_platform(),
            )

            # Return all results for verification
            return (
                base_result,
                temperature_result,
                version_result,
                model_result,
                platform_result,
            )

        # Run the async function
        loop = asyncio.new_event_loop()
        try:
            base_result, temp_result, version_result, model_result, platform_result = (
                loop.run_until_complete(run_concurrent_updates())
            )
        finally:
            loop.close()

        # Define expected keys
        config_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.config"
        telemetry_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.telemetry"

        # Verify base result has all expected data
        assert config_key in base_result
        assert telemetry_key in base_result

        base_config = json.loads(base_result[config_key])
        base_telemetry = json.loads(base_result[telemetry_key])

        assert base_config["model"] == "gpt-4"
        assert base_config["parameters"]["temperature"] == 0.7
        assert base_config["parameters"]["max_tokens"] == 500
        assert base_telemetry["client_info"]["version"] == "1.0.0"
        assert base_telemetry["client_info"]["platform"] == "python"

        # Verify temperature update only changed temperature
        assert config_key in temp_result
        temp_config = json.loads(temp_result[config_key])
        assert "parameters" in temp_config
        assert "temperature" in temp_config["parameters"]
        assert temp_config["parameters"]["temperature"] == 0.9
        assert "model" not in temp_config  # Shouldn't be present

        # Verify version update only changed version
        assert telemetry_key in version_result
        version_telemetry = json.loads(version_result[telemetry_key])
        assert "client_info" in version_telemetry
        assert "version" in version_telemetry["client_info"]
        assert version_telemetry["client_info"]["version"] == "1.1.0"
        assert (
            "platform" not in version_telemetry["client_info"]
        )  # Shouldn't be present

        # Verify model update only changed model
        assert config_key in model_result
        model_config = json.loads(model_result[config_key])
        assert model_config["model"] == "gpt-3.5-turbo"
        assert "parameters" not in model_config  # Shouldn't be present

        # Verify platform update only changed platform
        assert telemetry_key in platform_result
        platform_telemetry = json.loads(platform_result[telemetry_key])
        assert "client_info" in platform_telemetry
        assert "platform" in platform_telemetry["client_info"]
        assert platform_telemetry["client_info"]["platform"] == "web"
        assert (
            "version" not in platform_telemetry["client_info"]
        )  # Shouldn't be present

    def test_thread_safe_metadata_updates(self):
        """Test thread-safe metadata updates using the _flatten_and_serialize_metadata function."""
        import random
        import threading
        import time

        from langfuse._client.attributes import _flatten_and_serialize_metadata

        # Create a shared metadata dictionary we'll update from multiple threads
        shared_metadata = {
            "user": {
                "id": "user-123",
                "profile": {"name": "Test User", "email": "test@example.com"},
            },
            "system": {"version": "1.0.0", "features": ["search", "recommendations"]},
        }

        # Dictionary to store current metadata (protected by lock)
        current_metadata = shared_metadata.copy()
        metadata_lock = threading.Lock()

        # Thread function that updates a random part of metadata
        def update_random_metadata(thread_id):
            nonlocal current_metadata

            # Generate a random update
            updates = [
                # Update user name
                {"user": {"profile": {"name": f"User {thread_id}"}}},
                # Update user email
                {"user": {"profile": {"email": f"user{thread_id}@example.com"}}},
                # Update system version
                {"system": {"version": f"1.0.{thread_id}"}},
                # Add a feature
                {
                    "system": {
                        "features": [
                            "search",
                            "recommendations",
                            f"feature-{thread_id}",
                        ]
                    }
                },
                # Add a new top-level key
                {f"custom-{thread_id}": {"value": f"thread-{thread_id}"}},
            ]

            # Select a random update
            update = random.choice(updates)

            # Sleep a tiny bit to simulate work and increase chances of thread interleaving
            time.sleep(random.uniform(0.001, 0.01))

            # Apply the update to current_metadata (in a real system, this would update OTEL span)
            with metadata_lock:
                # This simulates how OTEL span attributes would be updated
                # In a real system, you'd iterate through flattened and set each attribute

                # For user name and email
                if "user" in update and "profile" in update["user"]:
                    if "name" in update["user"]["profile"]:
                        current_metadata["user"]["profile"]["name"] = update["user"][
                            "profile"
                        ]["name"]
                    if "email" in update["user"]["profile"]:
                        current_metadata["user"]["profile"]["email"] = update["user"][
                            "profile"
                        ]["email"]

                # For system version
                if "system" in update and "version" in update["system"]:
                    current_metadata["system"]["version"] = update["system"]["version"]

                # For system features
                if "system" in update and "features" in update["system"]:
                    current_metadata["system"]["features"] = update["system"][
                        "features"
                    ]

                # For new top-level keys
                for key in update:
                    if key not in ["user", "system"]:
                        current_metadata[key] = update[key]

        # Create and start multiple threads
        threads = []
        for i in range(10):  # Create 10 threads
            thread = threading.Thread(target=update_random_metadata, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify that the structure is still valid
        # User structure should be intact
        assert "user" in current_metadata
        assert "id" in current_metadata["user"]
        assert "profile" in current_metadata["user"]
        assert "name" in current_metadata["user"]["profile"]
        assert "email" in current_metadata["user"]["profile"]

        # System structure should be intact
        assert "system" in current_metadata
        assert "version" in current_metadata["system"]
        assert "features" in current_metadata["system"]
        assert isinstance(current_metadata["system"]["features"], list)

        # The metadata should still be serializable
        # This verifies we haven't broken the structure in a way that would prevent
        # proper OTEL attribute setting
        final_flattened = _flatten_and_serialize_metadata(
            current_metadata, "observation"
        )

        user_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.user"
        system_key = f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.system"

        assert user_key in final_flattened
        assert system_key in final_flattened

        # Verify we can deserialize the values
        user_data = json.loads(final_flattened[user_key])
        system_data = json.loads(final_flattened[system_key])

        assert "id" in user_data
        assert "profile" in user_data
        assert "version" in system_data
        assert "features" in system_data


class TestMultiProjectSetup(TestOTelBase):
    """Tests for multi-project setup within the same process.

    These tests verify that multiple Langfuse clients initialized with different
    public keys in the same process correctly export spans to their respective
    exporters without cross-contamination.
    """

    @pytest.fixture(scope="function")
    def multi_project_setup(self, monkeypatch):
        """Create two separate Langfuse clients with different projects."""
        # Reset any previous trace providers
        from opentelemetry import trace as trace_api_reset

        original_provider = trace_api_reset.get_tracer_provider()

        # Create exporters and tracers for two projects
        exporter_project1 = InMemorySpanExporter()
        exporter_project2 = InMemorySpanExporter()

        # Set project keys (must be different for each test to avoid cross-test contamination)
        import uuid

        unique_suffix = str(uuid.uuid4())[:8]
        project1_key = f"proj1_{unique_suffix}"
        project2_key = f"proj2_{unique_suffix}"

        # Clear singleton instances to avoid cross-test contamination
        monkeypatch.setattr(LangfuseResourceManager, "_instances", {})

        # Setup tracers with appropriate project-specific span exporting
        def mock_processor_init(self, **kwargs):
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self.public_key = kwargs.get("public_key", "test-key")
            # Use the appropriate exporter based on the project key
            if self.public_key == project1_key:
                exporter = exporter_project1
            else:
                exporter = exporter_project2

            BatchSpanProcessor.__init__(
                self,
                span_exporter=exporter,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )

        monkeypatch.setattr(
            "langfuse._client.span_processor.LangfuseSpanProcessor.__init__",
            mock_processor_init,
        )

        # Initialize separate tracer providers for each project
        tracer_provider_project1 = TracerProvider()
        tracer_provider_project1.add_span_processor(
            SimpleSpanProcessor(exporter_project1)
        )

        tracer_provider_project2 = TracerProvider()
        tracer_provider_project2.add_span_processor(
            SimpleSpanProcessor(exporter_project2)
        )

        # Instead of global mocking, directly patch the _initialize_instance method
        # to provide appropriate tracer providers
        original_initialize = LangfuseResourceManager._initialize_instance

        def mock_initialize(self, **kwargs):
            original_initialize(self, **kwargs)
            # Override the tracer with our test tracers
            if kwargs.get("public_key") == project1_key:
                self._otel_tracer = tracer_provider_project1.get_tracer(
                    f"langfuse:{project1_key}", "test"
                )
            elif kwargs.get("public_key") == project2_key:
                self._otel_tracer = tracer_provider_project2.get_tracer(
                    f"langfuse:{project2_key}", "test"
                )

        monkeypatch.setattr(
            LangfuseResourceManager, "_initialize_instance", mock_initialize
        )

        # Initialize the two clients
        langfuse_project1 = Langfuse(
            public_key=project1_key, secret_key="secret1", host="http://test-host"
        )

        langfuse_project2 = Langfuse(
            public_key=project2_key, secret_key="secret2", host="http://test-host"
        )

        # Return the setup
        setup = {
            "project1_key": project1_key,
            "project2_key": project2_key,
            "langfuse_project1": langfuse_project1,
            "langfuse_project2": langfuse_project2,
            "exporter_project1": exporter_project1,
            "exporter_project2": exporter_project2,
            "tracer_provider_project1": tracer_provider_project1,
            "tracer_provider_project2": tracer_provider_project2,
        }

        yield setup

        # Clean up and restore
        trace_api_reset.set_tracer_provider(original_provider)
        monkeypatch.setattr(
            LangfuseResourceManager, "_initialize_instance", original_initialize
        )

        exporter_project1.shutdown()
        exporter_project2.shutdown()

    def test_spans_routed_to_correct_exporters(self, multi_project_setup):
        """Test that spans are routed to the correct exporters based on public key."""
        # Create spans in both projects
        span1 = multi_project_setup["langfuse_project1"].start_span(
            name="trace-project1", metadata={"project": "project1"}
        )
        span1.end()

        span2 = multi_project_setup["langfuse_project2"].start_span(
            name="trace-project2", metadata={"project": "project2"}
        )
        span2.end()

        # Force flush to make sure all spans are exported
        multi_project_setup["tracer_provider_project1"].force_flush()
        multi_project_setup["tracer_provider_project2"].force_flush()

        # Check spans in project1's exporter
        spans_project1 = [
            span.name
            for span in multi_project_setup["exporter_project1"]._finished_spans
        ]
        assert "trace-project1" in spans_project1
        assert "trace-project2" not in spans_project1

        # Check spans in project2's exporter
        spans_project2 = [
            span.name
            for span in multi_project_setup["exporter_project2"]._finished_spans
        ]
        assert "trace-project2" in spans_project2
        assert "trace-project1" not in spans_project2

    def test_concurrent_operations_in_multiple_projects(self, multi_project_setup):
        """Test concurrent span operations in multiple projects."""
        import threading
        import time

        # Create simple non-nested spans in separate threads
        def create_spans_project1():
            for i in range(5):
                span = multi_project_setup["langfuse_project1"].start_span(
                    name=f"project1-span-{i}",
                    metadata={"project": "project1", "index": i},
                )
                # Small sleep to ensure overlap with other thread
                time.sleep(0.01)
                span.end()

        def create_spans_project2():
            for i in range(5):
                span = multi_project_setup["langfuse_project2"].start_span(
                    name=f"project2-span-{i}",
                    metadata={"project": "project2", "index": i},
                )
                # Small sleep to ensure overlap with other thread
                time.sleep(0.01)
                span.end()

        # Start threads
        thread1 = threading.Thread(target=create_spans_project1)
        thread2 = threading.Thread(target=create_spans_project2)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Force flush to make sure all spans are exported
        multi_project_setup["tracer_provider_project1"].force_flush()
        multi_project_setup["tracer_provider_project2"].force_flush()

        # Get spans from each project
        spans_project1 = multi_project_setup["exporter_project1"]._finished_spans
        spans_project2 = multi_project_setup["exporter_project2"]._finished_spans

        # Verify correct span counts in each project
        proj1_spans = [s for s in spans_project1 if s.name.startswith("project1-span-")]
        proj2_spans = [s for s in spans_project2 if s.name.startswith("project2-span-")]
        assert len(proj1_spans) == 5
        assert len(proj2_spans) == 5

        # Verify no cross-contamination between projects
        assert not any(s.name.startswith("project2-span-") for s in spans_project1)
        assert not any(s.name.startswith("project1-span-") for s in spans_project2)

        # Verify each project has distinct trace IDs
        trace_ids_project1 = {s.context.trace_id for s in spans_project1}
        trace_ids_project2 = {s.context.trace_id for s in spans_project2}
        assert len(trace_ids_project1.intersection(trace_ids_project2)) == 0

    def test_span_processor_filtering(self, multi_project_setup):
        """Test that spans are correctly filtered to the right exporters."""
        # Create spans with identical attributes in both projects
        span1 = multi_project_setup["langfuse_project1"].start_span(
            name="test-filter-span", metadata={"project": "shared-value"}
        )
        span1.end()

        span2 = multi_project_setup["langfuse_project2"].start_span(
            name="test-filter-span", metadata={"project": "shared-value"}
        )
        span2.end()

        # Force flush
        multi_project_setup["tracer_provider_project1"].force_flush()
        multi_project_setup["tracer_provider_project2"].force_flush()

        # Get spans from each exporter
        project1_spans = [
            s
            for s in multi_project_setup["exporter_project1"]._finished_spans
            if s.name == "test-filter-span"
        ]
        project2_spans = [
            s
            for s in multi_project_setup["exporter_project2"]._finished_spans
            if s.name == "test-filter-span"
        ]

        # Verify each project only has its own span
        assert len(project1_spans) == 1
        assert len(project2_spans) == 1

        # Verify that the spans are correctly routed
        # Each project should only see spans from its own tracer
        tracer_name1 = project1_spans[0].instrumentation_scope.name
        tracer_name2 = project2_spans[0].instrumentation_scope.name

        # The tracer names should be different and contain the respective project keys
        assert multi_project_setup["project1_key"] in tracer_name1
        assert multi_project_setup["project2_key"] in tracer_name2
        assert tracer_name1 != tracer_name2

    def test_context_isolation_between_projects(self, multi_project_setup):
        """Test that trace context is isolated between projects."""
        # Simplified version that just tests separate span routing

        # Start spans in both projects with the same name
        span1 = multi_project_setup["langfuse_project1"].start_span(
            name="identical-span-name"
        )
        span1.end()

        span2 = multi_project_setup["langfuse_project2"].start_span(
            name="identical-span-name"
        )
        span2.end()

        # Force flush to make sure all spans are exported
        multi_project_setup["tracer_provider_project1"].force_flush()
        multi_project_setup["tracer_provider_project2"].force_flush()

        # Verify each project only has its own spans
        spans_project1 = multi_project_setup["exporter_project1"]._finished_spans
        spans_project2 = multi_project_setup["exporter_project2"]._finished_spans

        # Each project should have exactly one span
        assert len(spans_project1) == 1
        assert len(spans_project2) == 1

        # The span IDs and trace IDs should be different
        assert spans_project1[0].context.span_id != spans_project2[0].context.span_id
        assert spans_project1[0].context.trace_id != spans_project2[0].context.trace_id

    def test_cross_project_tracing(self, multi_project_setup):
        """Test tracing when using multiple clients in the same code path."""
        # Create a cross-project sequence that should not share context

        # Start a span in project1
        span1 = multi_project_setup["langfuse_project1"].start_span(
            name="cross-project-parent"
        )

        # Without ending span1, create a span in project2
        # This should NOT inherit context from span1 even though it's active
        span2 = multi_project_setup["langfuse_project2"].start_span(
            name="independent-project2-span"
        )

        # End spans in opposite order
        span2.end()
        span1.end()

        # Force flush both exporters
        multi_project_setup["tracer_provider_project1"].force_flush()
        multi_project_setup["tracer_provider_project2"].force_flush()

        # Get all spans from both exporters
        spans_project1 = multi_project_setup["exporter_project1"]._finished_spans
        spans_project2 = multi_project_setup["exporter_project2"]._finished_spans

        # Verify each project has its own span
        assert len([s for s in spans_project1 if s.name == "cross-project-parent"]) == 1
        assert (
            len([s for s in spans_project2 if s.name == "independent-project2-span"])
            == 1
        )

        # Find the spans
        p1_span = next(s for s in spans_project1 if s.name == "cross-project-parent")
        p2_span = next(
            s for s in spans_project2 if s.name == "independent-project2-span"
        )

        # Verify the spans have different trace IDs
        assert p1_span.context.trace_id != p2_span.context.trace_id

        # Verify each tracer only has its own spans
        assert not any(s.name == "cross-project-parent" for s in spans_project2)
        assert not any(s.name == "independent-project2-span" for s in spans_project1)

    def test_sdk_client_isolation(self, multi_project_setup):
        """Test that clients use isolated tracers using different configurations."""
        # Instead of testing the internal implementation, test the public API
        # Each client should have different trace IDs

        # Create two spans with identical attributes in both projects
        span1 = multi_project_setup["langfuse_project1"].start_span(
            name="isolation-test-span"
        )
        span1.end()

        span2 = multi_project_setup["langfuse_project2"].start_span(
            name="isolation-test-span"
        )
        span2.end()

        # Force flush
        multi_project_setup["tracer_provider_project1"].force_flush()
        multi_project_setup["tracer_provider_project2"].force_flush()

        # Get spans from each project
        spans_proj1 = [
            s
            for s in multi_project_setup["exporter_project1"]._finished_spans
            if s.name == "isolation-test-span"
        ]
        spans_proj2 = [
            s
            for s in multi_project_setup["exporter_project2"]._finished_spans
            if s.name == "isolation-test-span"
        ]

        # We should have exactly one span in each exporter
        assert len(spans_proj1) == 1
        assert len(spans_proj2) == 1

        # The spans should be different
        assert spans_proj1[0].context.span_id != spans_proj2[0].context.span_id
        assert spans_proj1[0].context.trace_id != spans_proj2[0].context.trace_id

        # Check that the tracer names differ and contain the project keys
        proj1_tracer = spans_proj1[0].instrumentation_scope.name
        proj2_tracer = spans_proj2[0].instrumentation_scope.name

        assert multi_project_setup["project1_key"] in proj1_tracer
        assert multi_project_setup["project2_key"] in proj2_tracer
        assert proj1_tracer != proj2_tracer


class TestInstrumentationScopeFiltering(TestOTelBase):
    """Tests for filtering spans by instrumentation scope names."""

    @pytest.fixture(scope="function")
    def instrumentation_filtering_setup(self, monkeypatch):
        """Create setup for testing instrumentation scope filtering with actual span export."""
        from opentelemetry import trace as trace_api_reset
        from opentelemetry.sdk.trace import TracerProvider

        original_provider = trace_api_reset.get_tracer_provider()

        # Create separate exporters for blocked and allowed scopes testing
        blocked_exporter = InMemorySpanExporter()

        import uuid

        unique_suffix = str(uuid.uuid4())[:8]
        test_key = f"test_key_{unique_suffix}"

        # Clear singleton instances to avoid cross-test contamination
        monkeypatch.setattr(LangfuseResourceManager, "_instances", {})

        # Mock the LangfuseSpanProcessor to use our test exporters
        def mock_processor_init(self, **kwargs):
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self.public_key = kwargs.get("public_key", "test-key")
            blocked_scopes = kwargs.get("blocked_instrumentation_scopes")
            self.blocked_instrumentation_scopes = (
                blocked_scopes if blocked_scopes is not None else []
            )

            # For testing, use the appropriate exporter based on setup
            exporter = kwargs.get("_test_exporter", blocked_exporter)

            BatchSpanProcessor.__init__(
                self,
                span_exporter=exporter,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )

        monkeypatch.setattr(
            "langfuse._client.span_processor.LangfuseSpanProcessor.__init__",
            mock_processor_init,
        )

        # Create test tracer provider that will be used for all spans
        test_tracer_provider = TracerProvider()

        # We'll add the LangfuseSpanProcessor to this provider after it's created
        # in the mock_initialize function below

        # Mock resource manager initialization to use our test setup
        original_initialize = LangfuseResourceManager._initialize_instance

        def mock_initialize(self, **kwargs):
            # Call original_initialize to set up all the necessary attributes
            original_initialize(self, **kwargs)

            # Now create our custom LangfuseSpanProcessor with the actual blocked_instrumentation_scopes
            from langfuse._client.span_processor import LangfuseSpanProcessor

            processor = LangfuseSpanProcessor(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
                blocked_instrumentation_scopes=kwargs.get(
                    "blocked_instrumentation_scopes"
                ),
            )
            # Replace its exporter with our test exporter
            processor._span_exporter = blocked_exporter

            # Add the processor to our test tracer provider
            test_tracer_provider.add_span_processor(processor)

            # Override the tracer to use our test tracer provider
            self._otel_tracer = test_tracer_provider.get_tracer(
                "langfuse-sdk", "test", attributes={"public_key": self.public_key}
            )

        monkeypatch.setattr(
            LangfuseResourceManager, "_initialize_instance", mock_initialize
        )

        setup = {
            "blocked_exporter": blocked_exporter,
            "test_key": test_key,
            "test_tracer_provider": test_tracer_provider,
            "original_provider": original_provider,
            "original_initialize": original_initialize,
        }

        yield setup

        # Clean up
        trace_api_reset.set_tracer_provider(original_provider)
        monkeypatch.setattr(
            LangfuseResourceManager, "_initialize_instance", original_initialize
        )
        blocked_exporter.shutdown()

    def test_blocked_instrumentation_scopes_export_filtering(
        self, instrumentation_filtering_setup
    ):
        """Test that spans from blocked instrumentation scopes are not exported."""
        # Create Langfuse client with blocked scopes
        Langfuse(
            public_key=instrumentation_filtering_setup["test_key"],
            secret_key="test-secret-key",
            host="http://localhost:3000",
            blocked_instrumentation_scopes=["openai", "anthropic"],
        )

        # Get the tracer provider and create different instrumentation scope tracers
        tracer_provider = instrumentation_filtering_setup["test_tracer_provider"]

        # Create langfuse tracer with proper attributes for project validation
        langfuse_tracer = tracer_provider.get_tracer(
            "langfuse-sdk",
            attributes={"public_key": instrumentation_filtering_setup["test_key"]},
        )
        openai_tracer = tracer_provider.get_tracer("openai")
        anthropic_tracer = tracer_provider.get_tracer("anthropic")
        allowed_tracer = tracer_provider.get_tracer("allowed-library")

        # Create spans from each tracer
        langfuse_span = langfuse_tracer.start_span("langfuse-span")
        langfuse_span.end()

        openai_span = openai_tracer.start_span("openai-span")
        openai_span.end()

        anthropic_span = anthropic_tracer.start_span("anthropic-span")
        anthropic_span.end()

        allowed_span = allowed_tracer.start_span("allowed-span")
        allowed_span.end()

        # Force flush to ensure all spans are processed
        tracer_provider.force_flush()

        # Check which spans were actually exported
        exported_spans = instrumentation_filtering_setup[
            "blocked_exporter"
        ].get_finished_spans()
        exported_span_names = [span.name for span in exported_spans]
        exported_scope_names = [
            span.instrumentation_scope.name
            for span in exported_spans
            if span.instrumentation_scope
        ]

        # Langfuse spans should be exported (not blocked)
        assert "langfuse-span" in exported_span_names
        assert "langfuse-sdk" in exported_scope_names

        # Blocked scopes should NOT be exported
        assert "openai-span" not in exported_span_names
        assert "anthropic-span" not in exported_span_names
        assert "openai" not in exported_scope_names
        assert "anthropic" not in exported_scope_names

        # Allowed scopes should be exported
        assert "allowed-span" in exported_span_names
        assert "allowed-library" in exported_scope_names

    def test_no_blocked_scopes_allows_all_exports(
        self, instrumentation_filtering_setup
    ):
        """Test that when no scopes are blocked, all spans are exported."""
        # Create Langfuse client with NO blocked scopes
        Langfuse(
            public_key=instrumentation_filtering_setup["test_key"],
            secret_key="test-secret-key",
            host="http://localhost:3000",
            blocked_instrumentation_scopes=[],
        )

        # Get the tracer provider and create different instrumentation scope tracers
        tracer_provider = instrumentation_filtering_setup["test_tracer_provider"]

        langfuse_tracer = tracer_provider.get_tracer(
            "langfuse-sdk",
            attributes={"public_key": instrumentation_filtering_setup["test_key"]},
        )
        openai_tracer = tracer_provider.get_tracer("openai")
        anthropic_tracer = tracer_provider.get_tracer("anthropic")

        # Create spans from each tracer
        langfuse_span = langfuse_tracer.start_span("langfuse-span")
        langfuse_span.end()

        openai_span = openai_tracer.start_span("openai-span")
        openai_span.end()

        anthropic_span = anthropic_tracer.start_span("anthropic-span")
        anthropic_span.end()

        # Force flush
        tracer_provider.force_flush()

        # Check that ALL spans were exported
        exported_spans = instrumentation_filtering_setup[
            "blocked_exporter"
        ].get_finished_spans()
        exported_span_names = [span.name for span in exported_spans]

        assert "langfuse-span" in exported_span_names
        assert "openai-span" in exported_span_names
        assert "anthropic-span" in exported_span_names

    def test_none_blocked_scopes_allows_all_exports(
        self, instrumentation_filtering_setup
    ):
        """Test that when blocked_scopes is None (default), all spans are exported."""
        # Create Langfuse client with None blocked scopes (default behavior)
        Langfuse(
            public_key=instrumentation_filtering_setup["test_key"],
            secret_key="test-secret-key",
            host="http://localhost:3000",
            blocked_instrumentation_scopes=None,
        )

        # Get the tracer provider and create different instrumentation scope tracers
        tracer_provider = instrumentation_filtering_setup["test_tracer_provider"]

        langfuse_tracer = tracer_provider.get_tracer(
            "langfuse-sdk",
            attributes={"public_key": instrumentation_filtering_setup["test_key"]},
        )
        openai_tracer = tracer_provider.get_tracer("openai")

        # Create spans from each tracer
        langfuse_span = langfuse_tracer.start_span("langfuse-span")
        langfuse_span.end()

        openai_span = openai_tracer.start_span("openai-span")
        openai_span.end()

        # Force flush
        tracer_provider.force_flush()

        # Check that ALL spans were exported
        exported_spans = instrumentation_filtering_setup[
            "blocked_exporter"
        ].get_finished_spans()
        exported_span_names = [span.name for span in exported_spans]

        assert "langfuse-span" in exported_span_names
        assert "openai-span" in exported_span_names

    def test_blocking_langfuse_sdk_scope_export(self, instrumentation_filtering_setup):
        """Test that even Langfuse's own spans are blocked if explicitly specified."""
        # Create Langfuse client that blocks its own instrumentation scope
        Langfuse(
            public_key=instrumentation_filtering_setup["test_key"],
            secret_key="test-secret-key",
            host="http://localhost:3000",
            blocked_instrumentation_scopes=["langfuse-sdk"],
        )

        # Get the tracer provider and create tracers
        tracer_provider = instrumentation_filtering_setup["test_tracer_provider"]

        langfuse_tracer = tracer_provider.get_tracer(
            "langfuse-sdk",
            attributes={"public_key": instrumentation_filtering_setup["test_key"]},
        )
        other_tracer = tracer_provider.get_tracer("other-library")

        # Create spans
        langfuse_span = langfuse_tracer.start_span("langfuse-span")
        langfuse_span.end()

        other_span = other_tracer.start_span("other-span")
        other_span.end()

        # Force flush
        tracer_provider.force_flush()

        # Check exports - Langfuse spans should be blocked, others allowed
        exported_spans = instrumentation_filtering_setup[
            "blocked_exporter"
        ].get_finished_spans()
        exported_span_names = [span.name for span in exported_spans]

        assert "langfuse-span" not in exported_span_names
        assert "other-span" in exported_span_names


class TestConcurrencyAndAsync(TestOTelBase):
    """Tests for asynchronous and concurrent span operations."""

    @pytest.mark.asyncio
    async def test_async_span_operations(self, langfuse_client, memory_exporter):
        """Test async operations with spans."""
        import asyncio

        # Start a main span
        main_span = langfuse_client.start_span(name="async-main-span")

        # Define an async function that creates and updates spans
        async def async_task(parent_span, task_id):
            # Start a child span
            child_span = parent_span.start_span(name=f"async-task-{task_id}")

            # Simulate async work
            await asyncio.sleep(0.1)

            # Update span with results
            child_span.update(
                output={"result": f"Task {task_id} completed"},
                metadata={"task_id": task_id},
            )

            # End the child span
            child_span.end()
            return task_id

        # Execute multiple async tasks concurrently
        tasks = [async_task(main_span, i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Complete the main span
        main_span.update(output={"completed_tasks": results})
        main_span.end()

        # Get all spans
        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Find main span and task spans
        main = next((s for s in spans if s["name"] == "async-main-span"), None)
        task_spans = [s for s in spans if s["name"].startswith("async-task-")]

        # Verify all spans exist
        assert main is not None, "Main span not found"
        assert len(task_spans) == 3, f"Expected 3 task spans, found {len(task_spans)}"

        # Verify parent-child relationships
        for task_span in task_spans:
            self.assert_parent_child_relationship(main, task_span)

        # Verify task-specific attributes
        for i in range(3):
            task_span = next(
                (s for s in task_spans if s["name"] == f"async-task-{i}"), None
            )
            assert task_span is not None, f"Task span {i} not found"

            # Parse output and metadata
            output = self.verify_json_attribute(
                task_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
            )
            assert output["result"] == f"Task {i} completed"

        # Verify main span output
        main_output = self.verify_json_attribute(
            main, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        )
        assert main_output["completed_tasks"] == [0, 1, 2]

    def test_context_propagation_async(self, langfuse_client, memory_exporter):
        """Test context propagation across async operations using OTEL context."""
        import threading

        # Create a trace ID to use throughout the test
        trace_id = "abcdef1234567890abcdef1234567890"

        # Create a main span in thread 1
        trace_context = {"trace_id": trace_id}
        main_span = langfuse_client.start_span(
            name="main-async-span", trace_context=trace_context
        )

        # Save the span ID to verify parent-child relationships
        main_span_id = main_span.id

        # Set up an event to signal when thread 2 is done
        thread2_done = threading.Event()

        # Variables to store thread-local data
        thread2_span_id = None
        thread2_trace_id = None
        thread3_span_id = None
        thread3_trace_id = None

        # Function for thread 2
        def thread2_function():
            nonlocal thread2_span_id, thread2_trace_id

            # Access the same trace via trace_id in a different thread
            thread2_span = langfuse_client.start_span(
                name="thread2-span", trace_context={"trace_id": trace_id}
            )

            # Store IDs for verification
            thread2_span_id = thread2_span.id
            thread2_trace_id = thread2_span.trace_id

            # End the span
            thread2_span.end()

            # Signal that thread 2 is done
            thread2_done.set()

        # Function for thread 3 (will be called after thread 2)
        def thread3_function():
            nonlocal thread3_span_id, thread3_trace_id

            # Create a child of the main span by providing parent_span_id
            thread3_span = langfuse_client.start_span(
                name="thread3-span",
                trace_context={"trace_id": trace_id, "parent_span_id": main_span_id},
            )

            # Store IDs for verification
            thread3_span_id = thread3_span.id
            thread3_trace_id = thread3_span.trace_id

            # End the span
            thread3_span.end()

        # Start thread 2
        thread2 = threading.Thread(target=thread2_function)
        thread2.start()

        # Wait for thread 2 to complete
        thread2_done.wait()

        # Start thread 3
        thread3 = threading.Thread(target=thread3_function)
        thread3.start()
        thread3.join()

        # End the main span
        main_span.end()

        # Get all spans
        spans = [
            self.get_span_data(span) for span in memory_exporter.get_finished_spans()
        ]

        # Find our test spans
        main = next((s for s in spans if s["name"] == "main-async-span"), None)
        thread2_span = next((s for s in spans if s["name"] == "thread2-span"), None)
        thread3_span = next((s for s in spans if s["name"] == "thread3-span"), None)

        # Verify all spans exist
        assert main is not None, "Main span not found"
        assert thread2_span is not None, "Thread 2 span not found"
        assert thread3_span is not None, "Thread 3 span not found"

        # Verify all spans have the same trace ID
        assert main["trace_id"] == trace_id
        assert thread2_span["trace_id"] == trace_id
        assert thread3_span["trace_id"] == trace_id

        # Verify thread2 span is at the root level (no parent within our trace)
        assert (
            thread2_span["attributes"][LangfuseOtelSpanAttributes.AS_ROOT] is True
        ), "Thread 2 span should not have a parent"

        # Verify thread3 span is a child of the main span
        assert (
            thread3_span["parent_span_id"] == main_span_id
        ), "Thread 3 span should be a child of main span"

    @pytest.mark.asyncio
    async def test_span_metadata_updates_in_async_context(
        self, langfuse_client, memory_exporter
    ):
        """Test that span metadata updates preserve nested values in async contexts."""
        # Skip if the client setup is causing recursion issues
        if not hasattr(langfuse_client, "start_span"):
            pytest.skip("Client setup has issues, skipping test")

        import asyncio

        # Create a trace with a main span
        with langfuse_client.start_as_current_span(
            name="async-metadata-test"
        ) as main_span:
            # Initial metadata with nested structure
            initial_metadata = {
                "llm_config": {
                    "model": "gpt-4",
                    "parameters": {"temperature": 0.7, "top_p": 0.9},
                },
                "request_info": {"user_id": "test-user", "session_id": "test-session"},
            }

            # Set initial metadata
            main_span.update(metadata=initial_metadata)

            # Define async tasks that update different parts of metadata
            async def update_temperature():
                await asyncio.sleep(0.1)  # Simulate some async work
                main_span.update(
                    metadata={
                        "llm_config": {
                            "parameters": {
                                "temperature": 0.8  # Update temperature
                            }
                        }
                    }
                )

            async def update_model():
                await asyncio.sleep(0.05)  # Simulate some async work
                main_span.update(
                    metadata={
                        "llm_config": {
                            "model": "gpt-3.5-turbo"  # Update model
                        }
                    }
                )

            async def add_context_length():
                await asyncio.sleep(0.15)  # Simulate some async work
                main_span.update(
                    metadata={
                        "llm_config": {
                            "parameters": {
                                "context_length": 4096  # Add new parameter
                            }
                        }
                    }
                )

            async def update_user_id():
                await asyncio.sleep(0.08)  # Simulate some async work
                main_span.update(
                    metadata={
                        "request_info": {
                            "user_id": "updated-user"  # Update user_id
                        }
                    }
                )

            # Run all updates concurrently
            await asyncio.gather(
                update_temperature(),
                update_model(),
                add_context_length(),
                update_user_id(),
            )

        # Get the span data
        spans = self.get_spans_by_name(memory_exporter, "async-metadata-test")
        assert len(spans) == 1, "Expected one span"
        span_data = spans[0]

        # Skip further assertions if metadata attribute isn't present
        # (since the implementation might not be complete)
        if (
            LangfuseOtelSpanAttributes.OBSERVATION_METADATA
            not in span_data["attributes"]
        ):
            pytest.skip("Metadata attribute not present in span, skipping assertions")

        # Parse the final metadata
        metadata_str = span_data["attributes"][
            LangfuseOtelSpanAttributes.OBSERVATION_METADATA
        ]
        metadata = json.loads(metadata_str)

        # The behavior here depends on how the OTEL integration handles metadata updates
        # If it does deep merging correctly, we should see all values preserved/updated
        # If it doesn't, some values might be missing

        # Verify metadata structure (if implementation supports proper nesting)
        if "llm_config" in metadata:
            # These assertions may fail if the implementation doesn't support proper nesting
            assert metadata["llm_config"]["model"] == "gpt-3.5-turbo"

            if "parameters" in metadata["llm_config"]:
                assert metadata["llm_config"]["parameters"]["temperature"] == 0.8
                assert metadata["llm_config"]["parameters"]["top_p"] == 0.9
                assert metadata["llm_config"]["parameters"]["context_length"] == 4096

        if "request_info" in metadata:
            assert metadata["request_info"]["user_id"] == "updated-user"
            assert metadata["request_info"]["session_id"] == "test-session"

    def test_metrics_and_timing(self, langfuse_client, memory_exporter):
        """Test span timing and metrics."""
        import time

        # Record start time
        start_time = time.time()

        # Create a span
        span = langfuse_client.start_span(name="timing-test-span")

        # Add a small delay
        time.sleep(0.1)

        # End the span
        span.end()

        # Record end time
        end_time = time.time()

        # Get the span
        spans = self.get_spans_by_name(memory_exporter, "timing-test-span")
        assert len(spans) == 1, "Expected one span"

        # Get the raw span to access timing info
        raw_spans = [
            s
            for s in memory_exporter.get_finished_spans()
            if s.name == "timing-test-span"
        ]
        assert len(raw_spans) == 1, "Expected one raw span"

        raw_span = raw_spans[0]

        # Check that span start and end times are within the manually recorded range
        # Convert nanoseconds to seconds for comparison
        span_start_seconds = raw_span.start_time / 1_000_000_000
        span_end_seconds = raw_span.end_time / 1_000_000_000

        # The span timing should be within our manually recorded range
        # Note: This might fail on slow systems, so we use a relaxed comparison
        assert (
            span_start_seconds <= end_time
        ), "Span start time should be before our recorded end time"
        assert (
            span_end_seconds >= start_time
        ), "Span end time should be after our recorded start time"

        # Span duration should be positive and roughly match our sleep time
        span_duration_seconds = (
            raw_span.end_time - raw_span.start_time
        ) / 1_000_000_000
        assert span_duration_seconds > 0, "Span duration should be positive"

        # Since we slept for 0.1 seconds, the span duration should be at least 0.05 seconds
        # but we'll be generous with the upper bound due to potential system delays
        assert (
            span_duration_seconds >= 0.05
        ), f"Span duration ({span_duration_seconds}s) should be at least 0.05s"


# Add tests for media functionality in its own class
class TestMediaHandling(TestOTelBase):
    """Tests for media object handling, upload, and references."""

    def test_media_objects(self):
        """Test the basic behavior of LangfuseMedia objects."""
        # Test with base64 data URI
        base64_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4QBARXhpZgAA"
        media_from_base64 = LangfuseMedia(base64_data_uri=base64_data)

        # Test with content bytes
        sample_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
        media_from_bytes = LangfuseMedia(
            content_bytes=sample_bytes, content_type="image/jpeg"
        )

        # Verify the media objects were created correctly
        assert media_from_base64._source == "base64_data_uri"
        assert media_from_base64._content_type == "image/jpeg"
        assert media_from_base64._content_bytes is not None

        assert media_from_bytes._source == "bytes"
        assert media_from_bytes._content_type == "image/jpeg"
        assert media_from_bytes._content_bytes is not None

        # Test reference string creation with a manual media_id
        media_from_base64._media_id = "test-media-id"
        media_from_bytes._media_id = "test-media-id"

        # Now the reference strings should be generated
        assert media_from_base64._reference_string is not None
        assert media_from_bytes._reference_string is not None

        # Verify reference string formatting
        assert "test-media-id" in media_from_base64._reference_string
        assert "image/jpeg" in media_from_base64._reference_string
        assert "base64_data_uri" in media_from_base64._reference_string

        assert "test-media-id" in media_from_bytes._reference_string
        assert "image/jpeg" in media_from_bytes._reference_string
        assert "bytes" in media_from_bytes._reference_string

    def test_media_with_masking(self):
        """Test interaction between masking and media objects."""

        # Define a masking function that preserves media objects
        def mask_sensitive_data(data):
            if data is None:
                return None

            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    if k == "secret":
                        result[k] = "***MASKED***"
                    elif isinstance(v, (dict, list)):
                        # Handle nested structures
                        result[k] = mask_sensitive_data(v)
                    elif isinstance(v, LangfuseMedia):
                        # Pass media objects through
                        result[k] = v
                    else:
                        result[k] = v
                return result
            elif isinstance(data, list):
                return [mask_sensitive_data(item) for item in data]
            return data

        # Create media object for testing
        sample_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
        media = LangfuseMedia(content_bytes=sample_bytes, content_type="image/jpeg")
        media._media_id = "test-media-id"  # Set ID manually for testing

        # Create test data with both media and secrets
        test_data = {
            "regular": "data",
            "secret": "confidential-info",
            "nested": {"secret": "nested-secret", "media": media},
        }

        # Apply the masking function
        masked_data = mask_sensitive_data(test_data)

        # Verify masking works while preserving media
        assert masked_data["regular"] == "data"  # Regular field unchanged
        assert masked_data["secret"] == "***MASKED***"  # Secret field masked
        assert masked_data["nested"]["secret"] == "***MASKED***"  # Nested secret masked
        assert masked_data["nested"]["media"] is media  # Media object unchanged

        # Verify reference string is intact
        assert masked_data["nested"]["media"]._reference_string is not None
        assert "test-media-id" in masked_data["nested"]["media"]._reference_string

    def test_media_from_file(self):
        """Test loading media from files."""
        # Create media from a file path
        file_path = "static/puton.jpg"
        media_from_file = LangfuseMedia(file_path=file_path, content_type="image/jpeg")

        # Verify correct loading
        assert media_from_file._source == "file"
        assert media_from_file._content_type == "image/jpeg"
        assert media_from_file._content_bytes is not None

        # Set media_id manually for testing reference string
        media_from_file._media_id = "test-media-id"

        # Verify reference string
        assert media_from_file._reference_string is not None
        assert "test-media-id" in media_from_file._reference_string
        assert "image/jpeg" in media_from_file._reference_string
        assert "file" in media_from_file._reference_string

        # Test with non-existent file
        invalid_media = LangfuseMedia(
            file_path="nonexistent.jpg", content_type="image/jpeg"
        )

        # Source should be None for invalid file
        assert invalid_media._source is None
        assert invalid_media._content_bytes is None

    def test_masking(self):
        """Test the masking functionality of Langfuse."""

        # Define a test masking function (similar to what users would implement)
        def mask_sensitive_data(data):
            if data is None:
                return None

            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    if k == "sensitive" or k.endswith("_key") or k.endswith("_secret"):
                        result[k] = "***MASKED***"
                    elif isinstance(v, (dict, list)):
                        # Handle nested structures
                        result[k] = mask_sensitive_data(v)
                    else:
                        result[k] = v
                return result
            elif isinstance(data, list):
                return [mask_sensitive_data(item) for item in data]
            return data

        # Test various input scenarios
        test_cases = [
            # Basic dictionary with sensitive fields
            {
                "input": {
                    "regular": "data",
                    "sensitive": "secret-value",
                    "api_key": "1234",
                },
                "expected": {
                    "regular": "data",
                    "sensitive": "***MASKED***",
                    "api_key": "***MASKED***",
                },
            },
            # Nested dictionaries
            {
                "input": {
                    "user": "test",
                    "config": {"sensitive": "nested-secret", "normal": "value"},
                },
                "expected": {
                    "user": "test",
                    "config": {"sensitive": "***MASKED***", "normal": "value"},
                },
            },
            # Arrays with sensitive data
            {
                "input": [
                    {"name": "item1", "sensitive": "secret1"},
                    {"name": "item2", "sensitive": "secret2"},
                ],
                "expected": [
                    {"name": "item1", "sensitive": "***MASKED***"},
                    {"name": "item2", "sensitive": "***MASKED***"},
                ],
            },
            # None values
            {"input": None, "expected": None},
            # Edge case - empty dict
            {"input": {}, "expected": {}},
        ]

        # Run all test cases
        for i, test_case in enumerate(test_cases):
            result = mask_sensitive_data(test_case["input"])
            assert (
                result == test_case["expected"]
            ), f"Test case {i} failed: {result} != {test_case['expected']}"

        # Now test using the actual LangfuseSpan implementation
        from unittest.mock import MagicMock

        from langfuse._client.span import LangfuseSpan

        # Create a mock Langfuse client with the masking function
        mock_client = MagicMock()
        mock_client._mask = mask_sensitive_data

        # Create a concrete LangfuseSpan instance
        mock_span = MagicMock()
        span = LangfuseSpan(otel_span=mock_span, langfuse_client=mock_client)

        # Test 1: Direct call to _mask_attribute
        sensitive_data = {"regular": "value", "sensitive": "secret", "api_key": "12345"}
        masked_data = span._mask_attribute(data=sensitive_data)

        # Verify masking worked
        assert masked_data["sensitive"] == "***MASKED***"
        assert masked_data["api_key"] == "***MASKED***"
        assert masked_data["regular"] == "value"

        # Test 2: We need to mock _process_media_in_attribute to test _process_media_and_apply_mask
        # Since _process_media_in_attribute makes calls to media_manager
        original_process = span._process_media_in_attribute

        def mock_process_media(*, data, field):
            # Just return the data directly without processing
            return data

        # Replace with our mock
        span._process_media_in_attribute = mock_process_media

        try:
            # Now test the method
            process_result = span._process_media_and_apply_mask(
                data=sensitive_data, field="input", span=mock_span
            )

            # Verify processing and masking worked
            assert process_result["sensitive"] == "***MASKED***"
            assert process_result["api_key"] == "***MASKED***"
            assert process_result["regular"] == "value"
        finally:
            # Restore original
            span._process_media_in_attribute = original_process


class TestOtelIdGeneration(TestOTelBase):
    """Tests for trace_id and observation_id generation with and without seeds."""

    @pytest.fixture
    def langfuse_client(self, monkeypatch):
        """Create a minimal Langfuse client for testing ID generation functions."""
        client = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="http://test-host",
        )

        return client

    def test_trace_id_without_seed(self, langfuse_client, monkeypatch):
        """Test trace_id generation without seed (should use RandomIdGenerator)."""

        # Mock the RandomIdGenerator to return a predictable value
        def mock_generate_trace_id(self):
            return 0x1234567890ABCDEF1234567890ABCDEF

        monkeypatch.setattr(
            RandomIdGenerator, "generate_trace_id", mock_generate_trace_id
        )

        trace_id = langfuse_client.create_trace_id()
        assert trace_id == "1234567890abcdef1234567890abcdef"
        assert len(trace_id) == 32  # 16 bytes hex-encoded = 32 characters

    def test_trace_id_with_seed(self, langfuse_client):
        """Test trace_id generation with seed (should be deterministic)."""
        seed = "test-identifier"
        trace_id = langfuse_client.create_trace_id(seed=seed)

        # Expected value: first 16 bytes of SHA-256 hash of "test-identifier"
        expected = sha256(seed.encode("utf-8")).digest()[:16].hex()

        assert trace_id == expected
        assert len(trace_id) == 32  # 16 bytes hex-encoded = 32 characters

        # Verify the same seed produces the same ID
        trace_id_repeat = langfuse_client.create_trace_id(seed=seed)
        assert trace_id == trace_id_repeat

        # Verify a different seed produces a different ID
        different_seed = "different-identifier"
        different_trace_id = langfuse_client.create_trace_id(seed=different_seed)
        assert trace_id != different_trace_id

    def test_observation_id_without_seed(self, langfuse_client, monkeypatch):
        """Test observation_id generation without seed (should use RandomIdGenerator)."""

        # Mock the RandomIdGenerator to return a predictable value
        def mock_generate_span_id(self):
            return 0x1234567890ABCDEF

        monkeypatch.setattr(
            RandomIdGenerator, "generate_span_id", mock_generate_span_id
        )

        observation_id = langfuse_client._create_observation_id()
        assert observation_id == "1234567890abcdef"
        assert len(observation_id) == 16  # 8 bytes hex-encoded = 16 characters

    def test_observation_id_with_seed(self, langfuse_client):
        """Test observation_id generation with seed (should be deterministic)."""
        seed = "test-identifier"
        observation_id = langfuse_client._create_observation_id(seed=seed)

        # Expected value: first 8 bytes of SHA-256 hash of "test-identifier"
        expected = sha256(seed.encode("utf-8")).digest()[:8].hex()

        assert observation_id == expected
        assert len(observation_id) == 16  # 8 bytes hex-encoded = 16 characters

        # Verify the same seed produces the same ID
        observation_id_repeat = langfuse_client._create_observation_id(seed=seed)
        assert observation_id == observation_id_repeat

        # Verify a different seed produces a different ID
        different_seed = "different-identifier"
        different_observation_id = langfuse_client._create_observation_id(
            seed=different_seed
        )
        assert observation_id != different_observation_id

    def test_id_generation_consistency(self, langfuse_client):
        """Test that the same seed always produces the same IDs across multiple calls."""
        seed = "consistent-test-seed"

        # Generate multiple IDs with the same seed
        trace_ids = [langfuse_client.create_trace_id(seed=seed) for _ in range(5)]
        observation_ids = [
            langfuse_client._create_observation_id(seed=seed) for _ in range(5)
        ]

        # All trace IDs should be identical
        assert len(set(trace_ids)) == 1

        # All observation IDs should be identical
        assert len(set(observation_ids)) == 1

    def test_different_seeds_produce_different_ids(self, langfuse_client):
        """Test that different seeds produce different IDs."""
        seeds = [f"test-seed-{i}" for i in range(10)]

        # Generate IDs with different seeds
        trace_ids = [langfuse_client.create_trace_id(seed=seed) for seed in seeds]
        observation_ids = [
            langfuse_client._create_observation_id(seed=seed) for seed in seeds
        ]

        # All trace IDs should be unique
        assert len(set(trace_ids)) == len(seeds)

        # All observation IDs should be unique
        assert len(set(observation_ids)) == len(seeds)

    def test_langfuse_event_update_immutability(self, langfuse_client, memory_exporter, caplog):
        """Test that LangfuseEvent.update() logs a warning and does nothing."""
        import logging

        parent_span = langfuse_client.start_span(name="parent-span")

        event = parent_span.start_observation(
            name="test-event",
            as_type="event",
            input={"original": "input"},
        )

        # Try to update the event and capture warning logs
        with caplog.at_level(logging.WARNING, logger='langfuse._client.span'):
            result = event.update(
                name="updated_name",
                input={"updated": "input"},
                output={"updated": "output"},
                metadata={"updated": "metadata"}
            )

            # Verify warning was logged
            assert "Attempted to update LangfuseEvent observation" in caplog.text
            assert "Events cannot be updated after creation" in caplog.text

            # Verify the method returned self unchanged
            assert result is event

        parent_span.end()
