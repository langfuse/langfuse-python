from opentelemetry.sdk.trace import TracerProvider

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.span_processor import LangfuseSpanProcessor

PUBLIC_KEY = "test-public-key"
SECRET_KEY = "test-secret-key"
PROVIDER_SCOPE = "com.anthropic.sdk.python"
GEN_AI_ATTRIBUTES = {"gen_ai.provider.name": "anthropic"}


def _create_processor(memory_exporter, **kwargs):
    tracer_provider = TracerProvider()
    processor = LangfuseSpanProcessor(
        public_key=PUBLIC_KEY,
        secret_key=SECRET_KEY,
        base_url="http://test-host",
        span_exporter=memory_exporter,
        **kwargs,
    )
    tracer_provider.add_span_processor(processor)

    return tracer_provider, processor


def _langfuse_tracer(tracer_provider):
    return tracer_provider.get_tracer(
        LANGFUSE_TRACER_NAME,
        "test",
        attributes={"public_key": PUBLIC_KEY},
    )


def _exported_span_names(memory_exporter):
    return sorted(span.name for span in memory_exporter.get_finished_spans())


def _run_provider_call_inside(tracer_provider, observation_type):
    langfuse_tracer = _langfuse_tracer(tracer_provider)
    provider_tracer = tracer_provider.get_tracer(PROVIDER_SCOPE)

    with langfuse_tracer.start_as_current_span("parent") as parent:
        # The SDK sets the observation type after the span has started
        parent.set_attribute(
            LangfuseOtelSpanAttributes.OBSERVATION_TYPE, observation_type
        )
        with provider_tracer.start_as_current_span(
            "anthropic.messages.create", attributes=GEN_AI_ATTRIBUTES
        ):
            pass


def test_drops_provider_genai_span_nested_in_langfuse_generation(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(memory_exporter)

    _run_provider_call_inside(tracer_provider, "generation")
    processor.force_flush()

    assert _exported_span_names(memory_exporter) == ["parent"]
    assert processor._open_langfuse_spans_by_id == {}


def test_keeps_provider_genai_span_nested_in_langfuse_span(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)

    _run_provider_call_inside(tracer_provider, "span")
    processor.force_flush()

    assert _exported_span_names(memory_exporter) == [
        "anthropic.messages.create",
        "parent",
    ]


def test_keeps_provider_genai_span_without_langfuse_parent(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    provider_tracer = tracer_provider.get_tracer(PROVIDER_SCOPE)

    with provider_tracer.start_as_current_span(
        "anthropic.messages.create", attributes=GEN_AI_ATTRIBUTES
    ):
        pass
    processor.force_flush()

    assert _exported_span_names(memory_exporter) == ["anthropic.messages.create"]


def test_custom_should_export_span_keeps_nested_provider_genai_span(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(
        memory_exporter, should_export_span=lambda span: True
    )

    _run_provider_call_inside(tracer_provider, "generation")
    processor.force_flush()

    assert _exported_span_names(memory_exporter) == [
        "anthropic.messages.create",
        "parent",
    ]


def test_drops_provider_genai_span_nested_in_client_generation(
    langfuse_memory_client, memory_exporter
):
    provider_tracer = langfuse_memory_client._resources.tracer_provider.get_tracer(
        PROVIDER_SCOPE
    )

    with langfuse_memory_client.start_as_current_observation(
        name="ChatAnthropic", as_type="generation"
    ):
        with provider_tracer.start_as_current_span(
            "anthropic.messages.create", attributes=GEN_AI_ATTRIBUTES
        ):
            pass
    langfuse_memory_client.flush()

    assert _exported_span_names(memory_exporter) == ["ChatAnthropic"]
