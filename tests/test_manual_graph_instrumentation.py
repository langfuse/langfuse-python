import time

from langfuse import Langfuse, observe
from tests.utils import create_uuid, get_api


def test_observe_type_agent_instrumentation():
    """Test @observe(as_type='agent') with the as_type-based approach."""
    langfuse = Langfuse()
    api = get_api()

    trace_name = f"type_based_graph_test_{create_uuid()}"

    @observe(type="generation")
    def start_agent():
        time.sleep(0.1)
        return {"status": "started", "data": "initial_data"}

    @observe(type="retriever")
    def process_agent():
        time.sleep(0.1)
        return {"status": "processed", "data": "processed_data"}

    @observe(type="tool")
    def tool_call():
        time.sleep(0.1)
        return {"status": "intermediate", "data": "intermediate_data"}

    @observe(type="generation")
    def end_agent():
        time.sleep(0.1)
        return {"status": "completed", "data": "final_data"}

    # Run the workflow within a trace context
    with langfuse.start_as_current_span(name="agent_workflow", as_type="agent"):
        langfuse.update_current_trace(name=trace_name)

        start_result = start_agent()
        process_result = process_agent()
        tool_result = tool_call()
        end_result = end_agent()

        _workflow_result = {
            "start": start_result,
            "process": process_result,
            "tool": tool_result,
            "end": end_result,
        }

    langfuse.flush()
    time.sleep(0.5)

    traces = api.trace.list(limit=50)
    test_trace = None
    for i, trace_data in enumerate(traces.data):
        if trace_data.name == trace_name:
            test_trace = trace_data
            break

    assert test_trace is not None, f"Could not find trace with name {trace_name}"

    # Get the trace details including observations
    trace_details = api.trace.get(test_trace.id)
    all_observations = trace_details.observations

    agent_observations = [
        obs
        for obs in all_observations
        if obs.type in ["agent", "tool", "retriever", "chain", "embedding", "AGENT", "TOOL", "RETRIEVER", "CHAIN", "EMBEDDING", "GENERATION", "generation"]
    ]

    assert (
        len(agent_observations) == 4
    ), f"Expected 4 observations, got {len(agent_observations)} out of {len(all_observations)} total observations"

    # for agent_obs in agent_observations:
    #     print(
    #         f"{agent_obs.name} ({agent_obs.type}): {agent_obs.start_time} - {agent_obs.end_time}"
    #     )


def test_observe_type_parallel_tool_execution():
    """Test parallel tool execution where an agent starts multiple tools simultaneously.

    Creates a graph structure:
    start_agent -> [tool_1, tool_2, tool_3] -> end_agent
    """

    langfuse = Langfuse()
    api = get_api()

    trace_name = f"parallel_tools_test_{create_uuid()}"

    @observe(type="agent")
    def start_agent():
        time.sleep(0.05)
        return {"status": "tools_initiated", "tool_count": 3}

    @observe(type="tool")
    def search_tool():
        time.sleep(0.2)
        return {"tool": "search", "results": ["result1", "result2"]}

    @observe(type="tool")
    def calculation_tool():
        time.sleep(0.15)
        return {"tool": "calc", "result": 42}

    @observe(type="tool")
    def api_tool():
        time.sleep(0.1)
        return {"tool": "api", "data": {"status": "success"}}

    @observe(type="agent")
    def end_agent():
        time.sleep(0.05)
        return {"status": "completed", "summary": "all_tools_processed"}

    # Execute the parallel workflow
    with langfuse.start_as_current_span(name="parallel_workflow", as_type="span"):
        langfuse.update_current_trace(name=trace_name)
        start_result = start_agent()

        # execute tools in parallel - but keep them in the same trace context
        # we simulate parallel execution with staggered starts

        search_result = search_tool()
        time.sleep(0.01)
        calc_result = calculation_tool()
        time.sleep(0.01)
        api_result = api_tool()

        tool_results = {
            "search": search_result,
            "calculation": calc_result,
            "api": api_result,
        }

        end_result = end_agent()

        _workflow_result = {
            "start": start_result,
            "tools": tool_results,
            "end": end_result,
        }

    langfuse.flush()
    time.sleep(0.5)

    traces = api.trace.list(limit=50)
    test_trace = None

    for i, trace_data in enumerate(traces.data):
        if trace_data.name == trace_name:
            test_trace = trace_data
            break

    assert test_trace is not None, f"Could not find trace with name {trace_name}"

    # Get trace details and filter observations
    trace_details = api.trace.get(test_trace.id)
    all_observations = trace_details.observations

    graph_observations = [
        obs
        for obs in all_observations
        if obs.type in ["agent", "tool", "AGENT", "TOOL"]
    ]

    # Should have: start_agent (1) + 3 tools (3) + end_agent (1) = 5 total
    expected_count = 5
    assert (
        len(graph_observations) == expected_count
    ), f"Expected {expected_count} graph observations, got {len(graph_observations)} out of {len(all_observations)} total"

    # for obs in sorted(graph_observations, key=lambda x: x.start_time):
    #     print(f"   {obs.name} ({obs.type}): {obs.start_time} - {obs.end_time}")

    agent_observations = [obs for obs in graph_observations if obs.type == "agent"]
    tool_observations = [obs for obs in graph_observations if obs.type == "tool"]

    assert (
        len(agent_observations) == 2
    ), f"Expected 2 agent observations, got {len(agent_observations)}"
    assert (
        len(tool_observations) == 3
    ), f"Expected 3 tool observations, got {len(tool_observations)}"
