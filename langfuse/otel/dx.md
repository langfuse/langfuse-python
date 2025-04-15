# Desired DX

## Option 1: Langfuse classic

```python
from langfuse import langfuse

langfuse = Langfuse()

trace = langfuse.trace(id='trace-123', name='my_trace', tags=['tag1', 'tag2'], metadata={"key1": "val"})
span = langfuse.span(id='span-123', name='my-span')
generation = langfuse.generation(input='hello', output="what's up", usage_details={"input_tokens": 1, "output_tokens": 2})

generation.end()
span.end()
trace.end()
```

## Option 2: OTEL native, i.e. use spans throughout

```python
from langfuse import langfuse

langfuse = Langfuse()

root_span = langfuse.span(trace={"id": "new-trace-id", name: "new-trace-name"}, id="root-span-id")
regular_span = langfuse.span(id='span-123', name='my-span')
generation = langfuse.span(as_type="generation", input='hello', output="what's up", usage_details={"input_tokens": 1, "output_tokens": 2})

generation.end()
span.end()
trace.end()
```

## Late updates

```python
trace_1 = langfuse.trace()
trace_2 = langfuse.trace()

trace_1.span()
trace_2.span()

trace_1
```
