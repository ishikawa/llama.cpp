import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


def test_responses_stream_with_llama_telemetry():
    global server
    server.n_ctx = 256
    server.n_batch = 32
    server.n_slots = 1
    server.start()

    saw_progress = False
    saw_delta_timings = False
    completed = None

    res = server.make_stream_request("POST", "/responses", data={
        "input": "This is a test" * 10,
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    })

    for data in res:
        if "prompt_progress" in data:
            assert data["type"] == "response.in_progress"
            assert data["prompt_progress"]["total"] > 0
            assert data["prompt_progress"]["processed"] >= data["prompt_progress"]["cache"]
            saw_progress = True
        if "timings" in data:
            assert "prompt_per_second" in data["timings"]
            assert "predicted_per_second" in data["timings"]
            if data["type"] == "response.output_text.delta":
                saw_delta_timings = True
        if data["type"] == "response.completed":
            completed = data

    assert saw_progress
    assert saw_delta_timings
    assert completed is not None
    assert "usage" in completed["response"]
    assert "timings" in completed


@pytest.mark.slow
@pytest.mark.parametrize("stream", [False, True])
def test_responses_reasoning_summary(stream):
    """When the request carries reasoning.summary, the reasoning output item's
    'summary' array is populated with the raw thinking text (transcribed
    as a single summary_text part), and usage reports
    output_tokens_details.reasoning_tokens."""
    global server
    server = ServerProcess()
    server.model_hf_repo = "Qwen/Qwen3-1.7B-GGUF"
    server.model_hf_file = "Qwen3-1.7B-Q8_0.gguf"
    server.jinja = True
    server.n_ctx = 4096
    server.n_predict = 300
    server.server_port = 8085
    server.start(timeout_seconds=600)  # model needs time to download

    body = {
        "input": "What is 12*7? Think step by step, then answer.",
        "max_output_tokens": 300,
        "reasoning": {"summary": "auto"},
    }

    if not stream:
        res = server.make_request("POST", "/responses", data=body)
        assert res.status_code == 200
        reasoning_items = [item for item in res.body["output"] if item["type"] == "reasoning"]
        assert len(reasoning_items) == 1, f'Expected exactly one reasoning item, got {res.body["output"]}'
        assert reasoning_items[0]["status"] == "completed"
        summary = reasoning_items[0]["summary"]
        assert len(summary) == 1, f'Expected one summary part, got {summary}'
        assert summary[0]["type"] == "summary_text"
        assert len(summary[0]["text"]) > 0
        # the summary is a transcript of the raw thinking, so it matches the reasoning content
        assert summary[0]["text"] == reasoning_items[0]["content"][0]["text"]
        assert res.body["usage"]["output_tokens_details"]["reasoning_tokens"] > 0
    else:
        events = list(server.make_stream_request("POST", "/responses", data={**body, "stream": True}))

        summary_deltas = [e for e in events if e["type"] == "response.reasoning_summary_text.delta"]
        assert len(summary_deltas) > 0, "Should have response.reasoning_summary_text.delta events"

        summary_dones = [e for e in events if e["type"] == "response.reasoning_summary_text.done"]
        assert len(summary_dones) == 1, "Should have exactly one response.reasoning_summary_text.done event"

        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        response = completed[0]["response"]

        reasoning_items = [item for item in response["output"] if item["type"] == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["status"] == "completed"
        assert len(reasoning_items[0]["summary"]) == 1
        assert reasoning_items[0]["summary"][0]["type"] == "summary_text"
        assert response["usage"]["output_tokens_details"]["reasoning_tokens"] > 0


@pytest.mark.slow
def test_responses_reasoning_summary_not_requested():
    """Without reasoning.summary in the request, behavior is unchanged: the
    reasoning output item's 'summary' array stays empty. usage still reports
    output_tokens_details.reasoning_tokens, independent of whether a summary
    was requested."""
    global server
    server = ServerProcess()
    server.model_hf_repo = "Qwen/Qwen3-1.7B-GGUF"
    server.model_hf_file = "Qwen3-1.7B-Q8_0.gguf"
    server.jinja = True
    server.n_ctx = 4096
    server.n_predict = 300
    server.server_port = 8085
    server.start(timeout_seconds=600)  # model needs time to download

    res = server.make_request("POST", "/responses", data={
        "input": "What is 12*7? Think step by step, then answer.",
        "max_output_tokens": 300,
    })
    assert res.status_code == 200
    reasoning_items = [item for item in res.body["output"] if item["type"] == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["status"] == "completed"
    assert reasoning_items[0]["summary"] == []
    assert res.body["usage"]["output_tokens_details"]["reasoning_tokens"] > 0
