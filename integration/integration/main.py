from langfuse import Langfuse


def main():
    langfuse = Langfuse(
        host="http://localhost:3000",
        public_key="pk-lf-1234567890",
        secret_key="sk-lf-1234567890",
    )

    trace = langfuse.trace(name="trace-name")
    span = trace.span(name="span-name")
    generation = trace.generation(name="generation-name")
    generation.end()
    span.end()

    langfuse.flush()

    returned_trace = langfuse.get_trace(trace.id)

    assert returned_trace.id == trace.id
    assert returned_trace.name == "trace-name"


if __name__ == "__main__":
    main()
