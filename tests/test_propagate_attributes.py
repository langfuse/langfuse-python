"""Comprehensive tests for propagate_attributes functionality.

This module tests the propagate_attributes context manager that allows setting
trace-level attributes (user_id, session_id, metadata) that automatically propagate
to all child spans within the context.
"""

import concurrent.futures

import pytest
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from langfuse import propagate_attributes
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from tests.test_otel import TestOTelBase


class TestPropagateAttributesBase(TestOTelBase):
    """Base class for propagate_attributes tests with shared helper methods."""

    @pytest.fixture
    def langfuse_client(self, monkeypatch, tracer_provider, mock_processor_init):
        """Create a mocked Langfuse client with explicit tracer_provider for testing."""
        from langfuse import Langfuse

        # Set environment variables
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")

        # Create test client with explicit tracer_provider
        client = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="http://test-host",
            tracing_enabled=True,
            tracer_provider=tracer_provider,  # Pass the test provider explicitly
        )

        yield client

    def get_span_by_name(self, memory_exporter, name: str) -> dict:
        """Get single span by name (assert exactly one exists).

        Args:
            memory_exporter: The in-memory span exporter fixture
            name: The name of the span to retrieve

        Returns:
            dict: The span data as a dictionary

        Raises:
            AssertionError: If zero or more than one span with the name exists
        """
        spans = self.get_spans_by_name(memory_exporter, name)
        assert len(spans) == 1, f"Expected 1 span named '{name}', found {len(spans)}"
        return spans[0]

    def verify_missing_attribute(self, span_data: dict, attr_key: str):
        """Verify that a span does NOT have a specific attribute.

        Args:
            span_data: The span data dictionary
            attr_key: The attribute key to check for absence

        Raises:
            AssertionError: If the attribute exists on the span
        """
        attributes = span_data["attributes"]
        assert (
            attr_key not in attributes
        ), f"Attribute '{attr_key}' should NOT be on span '{span_data['name']}'"


class TestPropagateAttributesBasic(TestPropagateAttributesBase):
    """Tests for basic propagate_attributes functionality."""

    def test_user_id_propagates_to_child_spans(self, langfuse_client, memory_exporter):
        """Verify user_id propagates to all child spans within context."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="test_user_123"):
                child1 = langfuse_client.start_span(name="child-span-1")
                child1.end()

                child2 = langfuse_client.start_span(name="child-span-2")
                child2.end()

        # Verify both children have user_id
        child1_span = self.get_span_by_name(memory_exporter, "child-span-1")
        self.verify_span_attribute(
            child1_span,
            LangfuseOtelSpanAttributes.TRACE_USER_ID,
            "test_user_123",
        )

        child2_span = self.get_span_by_name(memory_exporter, "child-span-2")
        self.verify_span_attribute(
            child2_span,
            LangfuseOtelSpanAttributes.TRACE_USER_ID,
            "test_user_123",
        )

    def test_session_id_propagates_to_child_spans(
        self, langfuse_client, memory_exporter
    ):
        """Verify session_id propagates to all child spans within context."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(session_id="session_abc"):
                child1 = langfuse_client.start_span(name="child-span-1")
                child1.end()

                child2 = langfuse_client.start_span(name="child-span-2")
                child2.end()

        # Verify both children have session_id
        child1_span = self.get_span_by_name(memory_exporter, "child-span-1")
        self.verify_span_attribute(
            child1_span,
            LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
            "session_abc",
        )

        child2_span = self.get_span_by_name(memory_exporter, "child-span-2")
        self.verify_span_attribute(
            child2_span,
            LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
            "session_abc",
        )

    def test_metadata_propagates_to_child_spans(self, langfuse_client, memory_exporter):
        """Verify metadata propagates to all child spans within context."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(
                metadata={"experiment": "variant_a", "version": "1.0"}
            ):
                child1 = langfuse_client.start_span(name="child-span-1")
                child1.end()

                child2 = langfuse_client.start_span(name="child-span-2")
                child2.end()

        # Verify both children have metadata
        child1_span = self.get_span_by_name(memory_exporter, "child-span-1")
        self.verify_span_attribute(
            child1_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.experiment",
            "variant_a",
        )
        self.verify_span_attribute(
            child1_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.version",
            "1.0",
        )

        child2_span = self.get_span_by_name(memory_exporter, "child-span-2")
        self.verify_span_attribute(
            child2_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.experiment",
            "variant_a",
        )
        self.verify_span_attribute(
            child2_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.version",
            "1.0",
        )

    def test_all_attributes_propagate_together(self, langfuse_client, memory_exporter):
        """Verify user_id, session_id, and metadata all propagate together."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(
                user_id="user_123",
                session_id="session_abc",
                metadata={"experiment": "test", "env": "prod"},
            ):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child has all attributes
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_span_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )
        self.verify_span_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_SESSION_ID, "session_abc"
        )
        self.verify_span_attribute(
            child_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.experiment",
            "test",
        )
        self.verify_span_attribute(
            child_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.env",
            "prod",
        )


class TestPropagateAttributesHierarchy(TestPropagateAttributesBase):
    """Tests for propagation across span hierarchies."""

    def test_propagation_to_direct_children(self, langfuse_client, memory_exporter):
        """Verify attributes propagate to all direct children."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user_123"):
                child1 = langfuse_client.start_span(name="child-1")
                child1.end()

                child2 = langfuse_client.start_span(name="child-2")
                child2.end()

                child3 = langfuse_client.start_span(name="child-3")
                child3.end()

        # Verify all three children have user_id
        for i in range(1, 4):
            child_span = self.get_span_by_name(memory_exporter, f"child-{i}")
            self.verify_span_attribute(
                child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
            )

    def test_propagation_to_grandchildren(self, langfuse_client, memory_exporter):
        """Verify attributes propagate through multiple levels of nesting."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user_123", session_id="session_abc"):
                with langfuse_client.start_as_current_span(name="child-span"):
                    grandchild = langfuse_client.start_span(name="grandchild-span")
                    grandchild.end()

        # Verify all three levels have attributes
        parent_span = self.get_span_by_name(memory_exporter, "parent-span")
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        grandchild_span = self.get_span_by_name(memory_exporter, "grandchild-span")

        for span in [parent_span, child_span, grandchild_span]:
            self.verify_span_attribute(
                span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
            )
            self.verify_span_attribute(
                span, LangfuseOtelSpanAttributes.TRACE_SESSION_ID, "session_abc"
            )

    def test_propagation_across_observation_types(
        self, langfuse_client, memory_exporter
    ):
        """Verify attributes propagate to different observation types."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user_123"):
                # Create span
                span = langfuse_client.start_span(name="test-span")
                span.end()

                # Create generation
                generation = langfuse_client.start_observation(
                    as_type="generation", name="test-generation"
                )
                generation.end()

        # Verify both observation types have user_id
        span_data = self.get_span_by_name(memory_exporter, "test-span")
        self.verify_span_attribute(
            span_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )

        generation_data = self.get_span_by_name(memory_exporter, "test-generation")
        self.verify_span_attribute(
            generation_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )


class TestPropagateAttributesTiming(TestPropagateAttributesBase):
    """Critical tests for early vs late propagation timing."""

    def test_early_propagation_all_spans_covered(
        self, langfuse_client, memory_exporter
    ):
        """Verify setting attributes early covers all child spans."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            # Set attributes BEFORE creating any children
            with propagate_attributes(user_id="user_123"):
                child1 = langfuse_client.start_span(name="child-1")
                child1.end()

                child2 = langfuse_client.start_span(name="child-2")
                child2.end()

                child3 = langfuse_client.start_span(name="child-3")
                child3.end()

        # Verify ALL children have user_id
        for i in range(1, 4):
            child_span = self.get_span_by_name(memory_exporter, f"child-{i}")
            self.verify_span_attribute(
                child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
            )

    def test_late_propagation_only_future_spans_covered(
        self, langfuse_client, memory_exporter
    ):
        """Verify late propagation only affects spans created after context entry."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            # Create child1 BEFORE propagate_attributes
            child1 = langfuse_client.start_span(name="child-1")
            child1.end()

            # NOW set attributes
            with propagate_attributes(user_id="user_123"):
                # Create child2 AFTER propagate_attributes
                child2 = langfuse_client.start_span(name="child-2")
                child2.end()

        # Verify: child1 does NOT have user_id, child2 DOES
        child1_span = self.get_span_by_name(memory_exporter, "child-1")
        self.verify_missing_attribute(
            child1_span, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )

        child2_span = self.get_span_by_name(memory_exporter, "child-2")
        self.verify_span_attribute(
            child2_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )

    def test_current_span_gets_attributes(self, langfuse_client, memory_exporter):
        """Verify the currently active span gets attributes when propagate_attributes is called."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            # Call propagate_attributes while parent-span is active
            with propagate_attributes(user_id="user_123"):
                pass

        # Verify parent span itself has the attribute
        parent_span = self.get_span_by_name(memory_exporter, "parent-span")
        self.verify_span_attribute(
            parent_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )

    def test_spans_outside_context_unaffected(self, langfuse_client, memory_exporter):
        """Verify spans created outside context don't get attributes."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            # Span before context
            span1 = langfuse_client.start_span(name="span-1")
            span1.end()

            # Span inside context
            with propagate_attributes(user_id="user_123"):
                span2 = langfuse_client.start_span(name="span-2")
                span2.end()

            # Span after context
            span3 = langfuse_client.start_span(name="span-3")
            span3.end()

        # Verify: only span2 has user_id
        span1_data = self.get_span_by_name(memory_exporter, "span-1")
        self.verify_missing_attribute(
            span1_data, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )

        span2_data = self.get_span_by_name(memory_exporter, "span-2")
        self.verify_span_attribute(
            span2_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )

        span3_data = self.get_span_by_name(memory_exporter, "span-3")
        self.verify_missing_attribute(
            span3_data, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )


class TestPropagateAttributesValidation(TestPropagateAttributesBase):
    """Tests for validation of propagated attribute values."""

    def test_user_id_over_200_chars_dropped(self, langfuse_client, memory_exporter):
        """Verify user_id over 200 characters is dropped with warning."""
        long_user_id = "x" * 201

        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id=long_user_id):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child does NOT have user_id
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )

    def test_session_id_over_200_chars_dropped(self, langfuse_client, memory_exporter):
        """Verify session_id over 200 characters is dropped with warning."""
        long_session_id = "y" * 201

        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(session_id=long_session_id):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child does NOT have session_id
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_SESSION_ID
        )

    def test_metadata_value_over_200_chars_dropped(
        self, langfuse_client, memory_exporter
    ):
        """Verify metadata values over 200 characters are dropped with warning."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(metadata={"key": "z" * 201}):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child does NOT have metadata.key
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.key"
        )

    def test_exactly_200_chars_accepted(self, langfuse_client, memory_exporter):
        """Verify exactly 200 characters is accepted (boundary test)."""
        user_id_200 = "x" * 200

        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id=user_id_200):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child HAS user_id
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_span_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, user_id_200
        )

    def test_201_chars_rejected(self, langfuse_client, memory_exporter):
        """Verify 201 characters is rejected (boundary test)."""
        user_id_201 = "x" * 201

        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id=user_id_201):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child does NOT have user_id
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )

    def test_non_string_user_id_dropped(self, langfuse_client, memory_exporter):
        """Verify non-string user_id is dropped with warning."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id=12345):  # type: ignore
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify child does NOT have user_id
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )

    def test_mixed_valid_invalid_metadata(self, langfuse_client, memory_exporter):
        """Verify mixed valid/invalid metadata - valid entries kept, invalid dropped."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(
                metadata={
                    "valid_key": "valid_value",
                    "invalid_key": "x" * 201,  # Too long
                    "another_valid": "ok",
                }
            ):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Verify: valid keys present, invalid key absent
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_span_attribute(
            child_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.valid_key",
            "valid_value",
        )
        self.verify_span_attribute(
            child_span,
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.another_valid",
            "ok",
        )
        self.verify_missing_attribute(
            child_span, f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.invalid_key"
        )


class TestPropagateAttributesNesting(TestPropagateAttributesBase):
    """Tests for nested propagate_attributes contexts."""

    def test_nested_contexts_inner_overwrites(self, langfuse_client, memory_exporter):
        """Verify inner context overwrites outer context values."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user1"):
                # Create span in outer context
                span1 = langfuse_client.start_span(name="span-1")
                span1.end()

                # Inner context with different user_id
                with propagate_attributes(user_id="user2"):
                    span2 = langfuse_client.start_span(name="span-2")
                    span2.end()

        # Verify: span1 has user1, span2 has user2
        span1_data = self.get_span_by_name(memory_exporter, "span-1")
        self.verify_span_attribute(
            span1_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user1"
        )

        span2_data = self.get_span_by_name(memory_exporter, "span-2")
        self.verify_span_attribute(
            span2_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user2"
        )

    def test_after_inner_context_outer_restored(self, langfuse_client, memory_exporter):
        """Verify outer context is restored after exiting inner context."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user1"):
                # Span in outer context
                span1 = langfuse_client.start_span(name="span-1")
                span1.end()

                # Inner context
                with propagate_attributes(user_id="user2"):
                    span2 = langfuse_client.start_span(name="span-2")
                    span2.end()

                # Back to outer context
                span3 = langfuse_client.start_span(name="span-3")
                span3.end()

        # Verify: span1 and span3 have user1, span2 has user2
        span1_data = self.get_span_by_name(memory_exporter, "span-1")
        self.verify_span_attribute(
            span1_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user1"
        )

        span2_data = self.get_span_by_name(memory_exporter, "span-2")
        self.verify_span_attribute(
            span2_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user2"
        )

        span3_data = self.get_span_by_name(memory_exporter, "span-3")
        self.verify_span_attribute(
            span3_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user1"
        )

    def test_nested_different_attributes(self, langfuse_client, memory_exporter):
        """Verify nested contexts with different attributes merge correctly."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user1"):
                # Inner context adds session_id
                with propagate_attributes(session_id="session1"):
                    span = langfuse_client.start_span(name="span-1")
                    span.end()

        # Verify: span has BOTH user_id and session_id
        span_data = self.get_span_by_name(memory_exporter, "span-1")
        self.verify_span_attribute(
            span_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user1"
        )
        self.verify_span_attribute(
            span_data, LangfuseOtelSpanAttributes.TRACE_SESSION_ID, "session1"
        )


class TestPropagateAttributesEdgeCases(TestPropagateAttributesBase):
    """Tests for edge cases and unusual scenarios."""

    def test_propagate_attributes_with_no_args(self, langfuse_client, memory_exporter):
        """Verify calling propagate_attributes() with no args doesn't error."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes():
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Should not crash, spans created normally
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        assert child_span is not None

    def test_none_values_ignored(self, langfuse_client, memory_exporter):
        """Verify None values are ignored without error."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id=None, session_id=None, metadata=None):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Should not crash, no attributes set
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID
        )
        self.verify_missing_attribute(
            child_span, LangfuseOtelSpanAttributes.TRACE_SESSION_ID
        )

    def test_empty_metadata_dict(self, langfuse_client, memory_exporter):
        """Verify empty metadata dict doesn't cause errors."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(metadata={}):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # Should not crash, no metadata attributes set
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        assert child_span is not None

    def test_all_invalid_metadata_values(self, langfuse_client, memory_exporter):
        """Verify all invalid metadata values results in no metadata attributes."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(
                metadata={
                    "key1": "x" * 201,  # Too long
                    "key2": "y" * 201,  # Too long
                }
            ):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        # No metadata attributes should be set
        child_span = self.get_span_by_name(memory_exporter, "child-span")
        self.verify_missing_attribute(
            child_span, f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.key1"
        )
        self.verify_missing_attribute(
            child_span, f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.key2"
        )

    def test_propagate_with_no_active_span(self, langfuse_client, memory_exporter):
        """Verify propagate_attributes works even with no active span."""
        # Call propagate_attributes without creating a parent span first
        with propagate_attributes(user_id="user_123"):
            # Now create a span
            with langfuse_client.start_as_current_span(name="span-1"):
                pass

        # Should not crash, span should have user_id
        span_data = self.get_span_by_name(memory_exporter, "span-1")
        self.verify_span_attribute(
            span_data, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user_123"
        )


class TestPropagateAttributesFormat(TestPropagateAttributesBase):
    """Tests for correct attribute formatting and naming."""

    def test_user_id_uses_correct_attribute_name(
        self, langfuse_client, memory_exporter
    ):
        """Verify user_id uses the correct OTel attribute name."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(user_id="user_123"):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        child_span = self.get_span_by_name(memory_exporter, "child-span")
        # Verify the exact attribute key is used
        assert LangfuseOtelSpanAttributes.TRACE_USER_ID in child_span["attributes"]
        assert (
            child_span["attributes"][LangfuseOtelSpanAttributes.TRACE_USER_ID]
            == "user_123"
        )

    def test_session_id_uses_correct_attribute_name(
        self, langfuse_client, memory_exporter
    ):
        """Verify session_id uses the correct OTel attribute name."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(session_id="session_abc"):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        child_span = self.get_span_by_name(memory_exporter, "child-span")
        # Verify the exact attribute key is used
        assert LangfuseOtelSpanAttributes.TRACE_SESSION_ID in child_span["attributes"]
        assert (
            child_span["attributes"][LangfuseOtelSpanAttributes.TRACE_SESSION_ID]
            == "session_abc"
        )

    def test_metadata_keys_properly_prefixed(self, langfuse_client, memory_exporter):
        """Verify metadata keys are properly prefixed with TRACE_METADATA."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(
                metadata={"experiment": "A", "version": "1.0", "env": "prod"}
            ):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        child_span = self.get_span_by_name(memory_exporter, "child-span")
        attributes = child_span["attributes"]

        # Verify each metadata key is properly prefixed
        expected_keys = [
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.experiment",
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.version",
            f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.env",
        ]

        for key in expected_keys:
            assert key in attributes, f"Expected key '{key}' not found in attributes"

    def test_multiple_metadata_keys_independent(self, langfuse_client, memory_exporter):
        """Verify multiple metadata keys are stored as independent attributes."""
        with langfuse_client.start_as_current_span(name="parent-span"):
            with propagate_attributes(metadata={"k1": "v1", "k2": "v2", "k3": "v3"}):
                child = langfuse_client.start_span(name="child-span")
                child.end()

        child_span = self.get_span_by_name(memory_exporter, "child-span")
        attributes = child_span["attributes"]

        # Verify all three are separate attributes with correct values
        assert attributes[f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.k1"] == "v1"
        assert attributes[f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.k2"] == "v2"
        assert attributes[f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.k3"] == "v3"


class TestPropagateAttributesThreading(TestPropagateAttributesBase):
    """Tests for propagate_attributes with ThreadPoolExecutor."""

    @pytest.fixture(autouse=True)
    def instrument_threading(self):
        """Auto-instrument threading for all tests in this class."""
        instrumentor = ThreadingInstrumentor()
        instrumentor.instrument()
        yield
        instrumentor.uninstrument()

    def test_propagation_with_threadpoolexecutor(
        self, langfuse_client, memory_exporter
    ):
        """Verify attributes propagate from main thread to worker threads."""

        def worker_function(span_name: str):
            """Worker creates a span in thread pool."""
            span = langfuse_client.start_span(name=span_name)
            span.end()
            return span_name

        with langfuse_client.start_as_current_span(name="main-span"):
            with propagate_attributes(user_id="main_user", session_id="main_session"):
                # Execute work in thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [
                        executor.submit(worker_function, f"worker-span-{i}")
                        for i in range(3)
                    ]
                    concurrent.futures.wait(futures)

        # Verify all worker spans have propagated attributes
        for i in range(3):
            worker_span = self.get_span_by_name(memory_exporter, f"worker-span-{i}")
            self.verify_span_attribute(
                worker_span,
                LangfuseOtelSpanAttributes.TRACE_USER_ID,
                "main_user",
            )
            self.verify_span_attribute(
                worker_span,
                LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
                "main_session",
            )

    def test_propagation_isolated_between_threads(
        self, langfuse_client, memory_exporter
    ):
        """Verify each thread's context is isolated from others."""

        def create_trace_with_user(user_id: str):
            """Create a trace with specific user_id."""
            with langfuse_client.start_as_current_span(name=f"trace-{user_id}"):
                with propagate_attributes(user_id=user_id):
                    span = langfuse_client.start_span(name=f"span-{user_id}")
                    span.end()

        # Run two traces concurrently with different user_ids
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(create_trace_with_user, "user1")
            future2 = executor.submit(create_trace_with_user, "user2")
            concurrent.futures.wait([future1, future2])

        # Verify each trace has the correct user_id (no mixing)
        span1 = self.get_span_by_name(memory_exporter, "span-user1")
        self.verify_span_attribute(
            span1, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user1"
        )

        span2 = self.get_span_by_name(memory_exporter, "span-user2")
        self.verify_span_attribute(
            span2, LangfuseOtelSpanAttributes.TRACE_USER_ID, "user2"
        )

    def test_nested_propagation_across_thread_boundary(
        self, langfuse_client, memory_exporter
    ):
        """Verify nested spans across thread boundaries inherit attributes."""

        def worker_creates_child():
            """Worker thread creates a child span."""
            child = langfuse_client.start_span(name="worker-child-span")
            child.end()

        with langfuse_client.start_as_current_span(name="main-parent-span"):
            with propagate_attributes(user_id="main_user"):
                # Create span in main thread
                main_child = langfuse_client.start_span(name="main-child-span")
                main_child.end()

                # Create span in worker thread
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(worker_creates_child)
                    future.result()

        # Verify both spans (main and worker) have user_id
        main_child_span = self.get_span_by_name(memory_exporter, "main-child-span")
        self.verify_span_attribute(
            main_child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "main_user"
        )

        worker_child_span = self.get_span_by_name(memory_exporter, "worker-child-span")
        self.verify_span_attribute(
            worker_child_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "main_user"
        )

    def test_worker_thread_can_override_propagated_attrs(
        self, langfuse_client, memory_exporter
    ):
        """Verify worker thread can override propagated attributes."""

        def worker_overrides_user():
            """Worker thread sets its own user_id."""
            with propagate_attributes(user_id="worker_user"):
                span = langfuse_client.start_span(name="worker-span")
                span.end()

        with langfuse_client.start_as_current_span(name="main-span"):
            with propagate_attributes(user_id="main_user"):
                # Create span in main thread
                main_span = langfuse_client.start_span(name="main-child-span")
                main_span.end()

                # Worker overrides with its own user_id
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(worker_overrides_user)
                    future.result()

        # Verify: main span has main_user, worker span has worker_user
        main_child = self.get_span_by_name(memory_exporter, "main-child-span")
        self.verify_span_attribute(
            main_child, LangfuseOtelSpanAttributes.TRACE_USER_ID, "main_user"
        )

        worker_span = self.get_span_by_name(memory_exporter, "worker-span")
        self.verify_span_attribute(
            worker_span, LangfuseOtelSpanAttributes.TRACE_USER_ID, "worker_user"
        )

    def test_multiple_workers_with_same_propagated_context(
        self, langfuse_client, memory_exporter
    ):
        """Verify multiple workers all inherit same propagated context."""

        def worker_function(worker_id: int):
            """Worker creates a span."""
            span = langfuse_client.start_span(name=f"worker-{worker_id}")
            span.end()

        with langfuse_client.start_as_current_span(name="main-span"):
            with propagate_attributes(session_id="shared_session"):
                # Submit 5 workers
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(worker_function, i) for i in range(5)]
                    concurrent.futures.wait(futures)

        # Verify all 5 workers have same session_id
        for i in range(5):
            worker_span = self.get_span_by_name(memory_exporter, f"worker-{i}")
            self.verify_span_attribute(
                worker_span,
                LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
                "shared_session",
            )

    def test_concurrent_traces_with_different_attributes(
        self, langfuse_client, memory_exporter
    ):
        """Verify concurrent traces with different attributes don't mix."""

        def create_trace(trace_id: int):
            """Create a trace with unique user_id."""
            with langfuse_client.start_as_current_span(name=f"trace-{trace_id}"):
                with propagate_attributes(user_id=f"user_{trace_id}"):
                    span = langfuse_client.start_span(name=f"span-{trace_id}")
                    span.end()

        # Create 10 traces concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_trace, i) for i in range(10)]
            concurrent.futures.wait(futures)

        # Verify each trace has its correct user_id (no mixing)
        for i in range(10):
            span = self.get_span_by_name(memory_exporter, f"span-{i}")
            self.verify_span_attribute(
                span, LangfuseOtelSpanAttributes.TRACE_USER_ID, f"user_{i}"
            )

    def test_exception_in_worker_preserves_context(
        self, langfuse_client, memory_exporter
    ):
        """Verify exception in worker doesn't corrupt main thread context."""

        def worker_raises_exception():
            """Worker creates span then raises exception."""
            span = langfuse_client.start_span(name="worker-span")
            span.end()
            raise ValueError("Test exception")

        with langfuse_client.start_as_current_span(name="main-span"):
            with propagate_attributes(user_id="main_user"):
                # Create span before worker
                span1 = langfuse_client.start_span(name="span-before")
                span1.end()

                # Worker raises exception (catch it)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(worker_raises_exception)
                    try:
                        future.result()
                    except ValueError:
                        pass  # Expected

                # Create span after exception
                span2 = langfuse_client.start_span(name="span-after")
                span2.end()

        # Verify both main thread spans still have correct user_id
        span_before = self.get_span_by_name(memory_exporter, "span-before")
        self.verify_span_attribute(
            span_before, LangfuseOtelSpanAttributes.TRACE_USER_ID, "main_user"
        )

        span_after = self.get_span_by_name(memory_exporter, "span-after")
        self.verify_span_attribute(
            span_after, LangfuseOtelSpanAttributes.TRACE_USER_ID, "main_user"
        )
