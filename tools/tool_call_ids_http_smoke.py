"""Live check that a running Strata server shapes tool calls correctly.

Every tool call the server answers with must carry its own `call_id` and (on the Responses API) its own output-item
`id`, announced once, addressed consistently in every later event, with one `output_index` per item.  Clients that
key their bookkeeping on those ids - DSH composes `${call_id}|${id}` and joins an advertised call, the call it runs
and the result it reports on that key - can only pair a result with its call if the ids hold.  This asks a real
model for parallel reads and checks what the server actually sent.

Dry run by default; `--run` is what sends traffic.  Read-only: it never reconfigures or restarts the server, and it
only sends synthetic prompts.  Needs a GPU and a loaded model, unlike `tools/test_tool_call_ids_http_smoke.py`, which
checks the same rules offline against canned event streams.

    python tools/tool_call_ids_http_smoke.py --run
    python tools/tool_call_ids_http_smoke.py --url http://127.0.0.1:8080 --model <name> --run
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
import uuid

CALL_ID = re.compile(r"call_[0-9a-f]{24}")
ITEM_ID = re.compile(r"(fc|ctc)_[0-9a-f]{32}")

READ = {"name": "read", "description": "Reads a file and returns its contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file."}},
                       "required": ["path"]}}
RESPONSES_TOOLS = [{"type": "function", **READ}]
CHAT_TOOLS = [{"type": "function", "function": READ}]
ANTHROPIC_TOOLS = [{"name": READ["name"], "description": READ["description"],
                    "input_schema": READ["parameters"]}]

PARALLEL = "Read both files now, in parallel, in one assistant turn: call read for path a.txt and call read for " \
           "path b.txt. Make both calls; write no text of your own."
IDENTICAL = "Call read twice in one assistant turn, both times with path exactly a.txt. Two calls, identical " \
            "arguments, no text of your own."
THREE = "Read a.txt, b.txt and c.txt in parallel: three read calls in one assistant turn, nothing else."


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Client:
    def __init__(self, url, model):
        self.url = url.rstrip('/')
        self.model = model

    def post(self, path, body):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode()
            if "text/event-stream" in (response.headers.get("Content-Type") or ""):
                return response.status, events_of(raw)
            return response.status, json.loads(raw)

    def responses(self, prompt, stream=False, extra_input=()):
        body = {"model": self.model, "input": [{"role": "user", "content": prompt}, *extra_input],
                "tools": RESPONSES_TOOLS, "store": False}
        if stream:
            body["stream"] = True
        return self.post("/v1/responses", body)

    def chat(self, prompt, stream=False):
        return self.post("/v1/chat/completions", {"model": self.model, "stream": stream, "tools": CHAT_TOOLS,
                                                  "messages": [{"role": "user", "content": prompt}]})

    def anthropic(self, prompt, stream=False):
        body = {"model": self.model, "max_tokens": 512, "tools": ANTHROPIC_TOOLS,
                "messages": [{"role": "user", "content": prompt}]}
        return self.post("/v1/messages", {**body, "stream": stream})


def events_of(raw):
    out = []
    for block in raw.split("\n\n"):
        data = [line[len("data: "):] for line in block.splitlines() if line.startswith("data: ")]
        if data and data[0].startswith("{"):
            out.append(json.loads(data[0]))
    return out


# ------------------------------------------------------------------ the checks (pure: canned streams work too)
def responses_stream_checks(events):
    """The invariant, checked on one Responses event stream.  Returns (the calls, the problems found)."""
    problems = []
    added = [e for e in events if e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"]
    done = [e for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"]
    completed = next((e for e in events if e["type"] == "response.completed"), None)
    if completed is None:
        return [], ["the stream never reached response.completed"]
    whole = [o for o in completed["response"]["output"] if o["type"] == "function_call"]

    if len(added) != len(done):
        problems.append(f"{len(added)} calls announced, {len(done)} closed")
    if len({e["item"]["call_id"] for e in added}) != len(added):
        problems.append(f"two calls share a call_id: {[e['item']['call_id'] for e in added]}")
    if len({e["item"]["id"] for e in added}) != len(added):
        problems.append(f"two calls share an item id: {[e['item']['id'] for e in added]}")
    if len({f'{e["item"]["call_id"]}|{e["item"]["id"]}' for e in added}) != len(added):
        problems.append("a `${call_id}|${id}` key is not unique")
    if not all(CALL_ID.fullmatch(e["item"]["call_id"]) and ITEM_ID.fullmatch(e["item"]["id"]) for e in added):
        problems.append("an id is not the opaque call_/fc_ form")

    # every argument event belongs to the item its call was announced as
    announced = {e["item"]["id"]: e["item"] for e in added}
    deltas: dict[str, str] = {}
    for e in events:
        if e["type"] == "response.function_call_arguments.delta":
            if e["item_id"] not in announced:
                problems.append(f"an argument piece for {e['item_id']}, which was never announced")
            deltas[e["item_id"]] = deltas.get(e["item_id"], "") + e["delta"]
        elif e["type"] == "response.function_call_arguments.done":
            if e["item_id"] not in announced:
                problems.append(f"arguments closed on {e['item_id']}, which was never announced")
            elif e["name"] != announced[e["item_id"]]["name"]:
                problems.append(f"{e['item_id']} closed as {e['name']!r}, announced as "
                                f"{announced[e['item_id']]['name']!r}")
    for item in done:
        if deltas.get(item["item"]["id"], "") != item["item"]["arguments"]:
            problems.append(f"{item['item']['call_id']}: its deltas are {deltas.get(item['item']['id'])!r}, its "
                            f"item carries {item['item']['arguments']!r}")

    # one output_index per item, on every event that refers to an item
    seen: dict[str, set[int]] = {}
    for e in events:
        key = e.get("item_id") or (e.get("item") or {}).get("id")
        if key is None:
            continue
        if "output_index" not in e:
            problems.append(f"an event for {key} carries no output_index")
        else:
            seen.setdefault(key, set()).add(e["output_index"])
    for key, indexes in seen.items():
        if len(indexes) > 1:
            problems.append(f"{key} moves between output indices {sorted(indexes)}")
    if len({next(iter(v)) for v in seen.values()}) != len(seen):
        problems.append(f"two items share an output_index: { {k: next(iter(v)) for k, v in seen.items()} }")

    # the completed response describes the same calls with the same ids
    if {c["id"] for c in whole} != {e["item"]["id"] for e in added}:
        problems.append("response.completed carries different item ids than the announcements")
    if {c["call_id"] for c in whole} != {e["item"]["call_id"] for e in added}:
        problems.append("response.completed carries different call_ids than the announcements")
    if sorted((c["name"], c["arguments"]) for c in whole) != sorted((e["item"]["name"], e["item"]["arguments"])
                                                                    for e in done):
        problems.append("response.completed carries different names or arguments than output_item.done")
    return whole, problems


def chat_stream_checks(chunks):
    """The invariant on Chat Completions chunks: one id per call, announced once, one index per call."""
    problems = []
    by: dict[int, dict] = {}
    first_at: dict[str, int] = {}
    owner: dict[int, str] = {}
    for chunk in chunks:
        for tc in chunk["choices"][0]["delta"].get("tool_calls") or []:
            index = tc.get("index", len(by))
            if tc.get("id"):
                if tc["id"] in first_at:
                    problems.append(f"{tc['id']} announced twice")
                first_at[tc["id"]] = index
                if index in owner:
                    problems.append(f"index {index} carries two calls, {owner[index]} and {tc['id']}")
                owner[index] = tc["id"]
            elif index not in owner:
                problems.append(f"arguments on index {index}, which no call opened")
            cur = by.setdefault(index, {"id": None, "name": "", "arguments": ""})
            cur["id"] = tc.get("id") or cur["id"]
            cur["name"] += (tc.get("function") or {}).get("name") or ""
            cur["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    calls = [by[i] for i in sorted(by)]
    if len({c["id"] for c in calls}) != len(calls):
        problems.append(f"two calls share an id: {[c['id'] for c in calls]}")
    if not all(CALL_ID.fullmatch(c["id"] or "") for c in calls):
        problems.append("an id is not the opaque call_ form")
    return calls, problems


def anthropic_stream_checks(events):
    """The invariant on /v1/messages events: one tool_use block per call, its own id, its own input."""
    problems = []
    blocks: dict[int, dict] = {}
    ids: dict[str, int] = {}
    for e in events:
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
            block = e["content_block"]
            if block["id"] in ids:
                problems.append(f"{block['id']} opened as two blocks")
            ids[block["id"]] = e["index"]
            blocks[e["index"]] = {"id": block["id"], "name": block["name"], "json": ""}
        elif e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
            if e["index"] not in blocks:
                problems.append(f"input for block {e['index']}, which was never opened")
            else:
                blocks[e["index"]]["json"] += e["delta"]["partial_json"]
    calls = [blocks[i] for i in sorted(blocks)]
    if len({c["id"] for c in calls}) != len(calls):
        problems.append(f"two blocks share an id: {[c['id'] for c in calls]}")
    if not all(CALL_ID.fullmatch(c["id"]) for c in calls):
        problems.append("a block id is not the opaque call_ form")
    return calls, problems


def canon(text):
    """Arguments as they compare across layers: JSON re-dumped compactly, so a layer that re-serialises
    (Anthropic's tool_use input, the whole-answer path) still matches the bytes the model wrote."""
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        return text


def whole_checks(message_calls, want):
    """A non-streamed answer must describe the same calls as the stream (ids are per request, so only the
    calls themselves are comparable across two requests)."""
    problems = []
    got = sorted((c["function"]["name"], canon(c["function"]["arguments"])) for c in message_calls)
    if got != sorted((c["name"], canon(c["arguments"])) for c in want):
        problems.append(f"the whole answer describes {got}, the stream described {want}")
    if len({c["id"] for c in message_calls}) != len(message_calls):
        problems.append(f"the whole answer reuses an id: {[c['id'] for c in message_calls]}")
    return problems


# ------------------------------------------------------------------ the run
def run(client):
    checks = []

    def check(name, problems, detail=""):
        checks.append({"name": name, "ok": not problems, "detail": detail or "; ".join(problems)})
        return not problems

    # parallel reads on the Responses API, streamed: the shape DSH keys its state on
    code, events = client.responses(PARALLEL, stream=True)
    require(code == 200, f"/v1/responses answered {code}: {json.dumps(events)[:200]}")
    calls, problems = responses_stream_checks(events)
    check("responses stream: parallel reads", problems,
          f"{len(calls)} calls " + str([(c["call_id"], c["arguments"]) for c in calls]))
    require(len(calls) >= 2, f"the model issued {len(calls)} calls; need at least 2 to check parallel shape")

    # the same request without streaming
    code, whole = client.responses(PARALLEL)
    require(code == 200, f"/v1/responses answered {code}")
    wcalls = [o for o in whole["output"] if o["type"] == "function_call"]
    problems = []
    if len({c["call_id"] for c in wcalls}) != len(wcalls) or len({c["id"] for c in wcalls}) != len(wcalls):
        problems.append(f"the whole answer reuses an id: {[(c['call_id'], c['id']) for c in wcalls]}")
    if sorted((c["name"], canon(c["arguments"])) for c in wcalls) != sorted(
            (c["name"], canon(c["arguments"])) for c in calls):
        problems.append(f"the whole answer describes {[(c['name'], c['arguments']) for c in wcalls]}, the stream "
                        f"{[(c['name'], c['arguments']) for c in calls]}")
    check("responses whole: the same calls", problems, f"{len(wcalls)} calls")

    # Chat Completions and Anthropic, streamed and whole
    code, chunks = client.chat(PARALLEL, stream=True)
    require(code == 200, f"/v1/chat/completions answered {code}")
    chat_calls, problems = chat_stream_checks(chunks)
    check("chat stream: parallel reads", problems, f"{len(chat_calls)} calls "
          + str([(c["id"], c["arguments"]) for c in chat_calls]))
    code, whole = client.chat(PARALLEL)
    require(code == 200, f"/v1/chat/completions answered {code}")
    message = whole["choices"][0]["message"]
    check("chat whole: the same calls", whole_checks(message.get("tool_calls") or [], chat_calls)
          + ([] if whole["choices"][0]["finish_reason"] == "tool_calls"
             else [f"finish_reason is {whole['choices'][0]['finish_reason']!r}, not 'tool_calls'"]),
          f"{len(message.get('tool_calls') or [])} calls")

    code, events = client.anthropic(PARALLEL, stream=True)
    require(code == 200, f"/v1/messages answered {code}")
    use, problems = anthropic_stream_checks(events)
    check("anthropic stream: parallel reads", problems, f"{len(use)} blocks "
          + str([(c["id"], c["json"]) for c in use]))
    code, whole = client.anthropic(PARALLEL)
    require(code == 200, f"/v1/messages answered {code}")
    blocks = [b for b in whole["content"] if b["type"] == "tool_use"]
    problems = whole_checks([{"id": b["id"], "function": {"name": b["name"],
                                                           "arguments": json.dumps(b["input"], ensure_ascii=False)}}
                            for b in blocks],
                            [{"name": c["name"], "arguments": c["json"]} for c in use])
    if len({b["id"] for b in blocks}) != len(blocks):
        problems.append(f"the whole message reuses a block id: {[b['id'] for b in blocks]}")
    check("anthropic whole: the same calls", problems, f"{len(blocks)} blocks")

    # two calls of the same tool with byte-identical arguments: the ids must still differ
    code, events = client.responses(IDENTICAL, stream=True)
    require(code == 200, f"/v1/responses answered {code}")
    same, problems = responses_stream_checks(events)
    check("identical arguments: distinct ids", problems, f"{len(same)} calls "
          + str([(c["call_id"], c["arguments"]) for c in same]))
    if same and len({c["arguments"] for c in same}) > 1:
        print("  note: the model did not repeat the same arguments, so this run did not test that case")
    elif same:
        check("identical arguments: the model really repeated them", [],
              f"arguments {json.dumps([c['arguments'] for c in same])}")

    # three parallel calls, replayed with results reported in the reverse of the call order
    code, events = client.responses(THREE, stream=True)
    require(code == 200, f"/v1/responses answered {code}")
    three, problems = responses_stream_checks(events)
    check("three parallel calls", problems, f"{len(three)} calls "
          + str([(c["call_id"], c["arguments"]) for c in three]))
    if len(three) >= 3:
        want = {c["call_id"]: f"RESULT-{len(three) - 1 - i}" for i, c in enumerate(three)}
        outs = [{"type": "function_call_output", "call_id": c["call_id"], "output": want[c["call_id"]]}
                for c in reversed(three)]
        ask = {"role": "user", "content": "Answer with one line per tool result you were given, in the order of the "
                                          "calls you made. Each line: <path> <result>. Use only the paths and results "
                                          "from this conversation. No other text."}
        code, back = client.responses("Read a.txt, b.txt and c.txt in parallel.", extra_input=[*three, *outs, ask])
        require(code == 200, f"the replayed turn answered {code}")
        text = "".join(p.get("text", "") for o in back["output"] if o["type"] == "message" for p in o["content"])
        pairs = re.findall(r"(a|b|c)\.txt\s+(RESULT-\d)", text)
        got = {canon(f'{{"path":"{p}.txt"}}'): r for p, r in pairs}
        problems = ([] if all(got.get(canon(c["arguments"])) == want[c["call_id"]] for c in three)
                    else [f"expected {sorted(want.values())}, the model paired {pairs}"])
        check("replay: each result reached the call its call_id named", problems, text[:200])

    # a result naming no call is refused rather than attached to some other call
    marker = "call_" + uuid.uuid4().hex[:24]
    try:
        code, answer = client.responses(PARALLEL, extra_input=[*calls,
                                                               {"type": "function_call_output", "call_id": marker,
                                                                "output": "orphan"}])
        problems = [f"an output for the unknown {marker} answered {code} instead of 400"]
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode())
        problems = ([] if error.code == 400 and (body.get("error") or {}).get("param", "").endswith("call_id")
                    else [f"HTTP {error.code}: {json.dumps(body)[:200]}"])
    check("an unknown call_id is refused with the field in param", problems, marker)

    return {"checks": checks, "model": client.model}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None, help="default: the model the server reports at /health")
    ap.add_argument("--run", action="store_true", help="send the requests (without it: nothing is sent)")
    args = ap.parse_args()
    if not args.run:
        print("Dry run: parallel reads, identical arguments, three calls replayed in reverse, an unknown call_id, "
              "on the Responses, Chat Completions and Anthropic APIs. No network traffic.")
        return
    client = Client(args.url, args.model or health_model(args.url))
    print(f"checking {client.url} ({client.model})")
    result = run(client)
    failed = [c for c in result["checks"] if not c["ok"]]
    for c in result["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']}  {c['detail']}")
    if failed:
        raise SystemExit(f"FAIL: {len(failed)} of {len(result['checks'])} checks")
    print(f"PASS: {len(result['checks'])} checks on {client.model}")


def health_model(url):
    with urllib.request.urlopen(url.rstrip('/') + "/health", timeout=20) as response:
        return json.load(response)["model"]


if __name__ == "__main__":
    main()
