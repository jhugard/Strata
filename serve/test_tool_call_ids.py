"""serve/test_tool_call_ids.py - tool-call ids are an enforced invariant, not a side effect of how they happen to
be generated.

Every `call_id` and every output-item `id` this server issues comes from one per-request `CallIds` registry, which is
seeded with the ids the incoming conversation already used, and every streamed event is addressed to the item its
call id names.  These tests pin that: distinct ids for parallel calls (even two calls with the same name and the same
arguments), one id per call across every event of its lifetime, argument pieces that never land on another call's
item, one `output_index` per item, streamed and whole answers that agree, results matched to the call their id names,
no id reissued by a later turn, and an unfinished call that keeps its own id.

Runs without a GPU or a model download (the mock engine, the parser and the three event layers directly).

    python -m unittest serve.test_tool_call_ids -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import re
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (CallIds, ChatTemplate, OutputParser, ToolCallPairingError, anthropic_to_messages,  # noqa: E402
                            openai_to_messages, pair_tool_results, parse_tool_call)
from serve.responses import Assembler, ResponsesError, collect, input_messages, request_tools  # noqa: E402
from serve.server import (ByteTokenizer, IM_END, MockEngine, Service, anthropic_collect,  # noqa: E402
                          anthropic_events, openai_chunks, openai_collect, serve)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
CTX = 8192
ID_RE = re.compile(r"call_[0-9a-f]{24}")            # the single-call format, unchanged: opaque hex, nothing of the
#                                                  call in it

PROPS = {"path": {"type": "string"}, "content": {"type": "string"}}
TOOL = {"name": "write", "description": "Writes a file.", "parameters": {"type": "object", "properties": PROPS}}
OPENAI_TOOLS = [{"type": "function", "function": TOOL}]
ANTHROPIC_TOOLS = [{"name": "write", "description": "Writes a file.", "input_schema": TOOL["parameters"]}]
RESP_TOOLS = [{"type": "function", "name": "write", "description": "Writes a file.",
               "parameters": TOOL["parameters"]}]
CUSTOM_TOOLS = [*RESP_TOOLS, {"type": "custom", "name": "apply_patch", "description": "Edits files."}]


def call(name: str, **args) -> str:
    """One tool call as the model writes it."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>\n"


TWO = ("Thinking.\n</think>\n\n" + call("write", path="a.txt", content="same")
       + call("write", path="a.txt", content="same"))
THREE = TWO + call("write", path="b.txt", content="other")
MIXED = ("Thinking.\n</think>\n\n" + call("write", path="a.txt", content="text")
         + call("apply_patch", input="*** Begin Patch\n"))
CUT = ("Thinking.\n</think>\n\n" + call("write", path="a.txt", content="whole")
       + "<tool_call>\n<function=write>\n<parameter=path>\nb.txt\n</parameter>\n<parameter=content>\nhal")
ANSWER = "Done.\n</think>\n\nBoth written."

ITEM_EVENTS = ("response.output_item.added", "response.output_item.done", "response.content_part.added",
               "response.content_part.done", "response.output_text.delta", "response.output_text.done",
               "response.reasoning_text.delta", "response.reasoning_text.done",
               "response.function_call_arguments.delta", "response.function_call_arguments.done",
               "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done")
CALL_EVENTS = ("response.output_item.added", "response.output_item.done", "response.function_call_arguments.delta",
               "response.function_call_arguments.done", "response.custom_tool_call_input.delta",
               "response.custom_tool_call_input.done")


def service(script: str, tok=None):
    tok = tok or ByteTokenizer()
    return tok, Service(MockEngine(tok, script, max_context=CTX), tok, TEMPLATE)


def parser_events(text: str, step: int, tools=None, ids=None) -> list:
    """Feed the model's text to OutputParser in pieces of `step` characters."""
    p = OutputParser(thinking=True, tools=tools or [TOOL], stream_tools=True, ids=ids)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    return evs + p.finish()


# ------------------------------------------------------------------ the parser: every call id of one answer is its own
class ParserIds(unittest.TestCase):
    def test_two_and_three_calls_get_distinct_ids(self):
        for text, want in ((TWO, 2), (THREE, 3)):
            for step in (1, 7, 10_000):
                with self.subTest(calls=want, step=step):
                    evs = parser_events(text, step)
                    started = [e.call for e in evs if e.kind == "tool_start"]
                    done = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(started), want)
                    self.assertEqual(len(done), want)
                    ids = [c.id for c in started]
                    self.assertEqual(len(set(ids)), want, ids)
                    self.assertEqual(ids, [c.id for c in done])       # the announced id is the final one
                    self.assertTrue(all(ID_RE.fullmatch(i) for i in ids), ids)
                    # the two identical calls really are identical: same name, same arguments
                    self.assertEqual((done[0].name, done[0].arguments), (done[1].name, done[1].arguments))

    def test_an_id_says_nothing_about_the_call(self):
        for ev in parser_events(THREE, 1):
            if ev.kind in ("tool_start", "tool_call"):
                self.assertTrue(ID_RE.fullmatch(ev.call.id))
                for part in (ev.call.name, "path", "a.txt", "b.txt"):
                    self.assertNotIn(part, ev.call.id)
                self.assertTrue(ev.call.id.isascii())

    def test_each_argument_piece_is_attributed_to_its_own_call(self):
        for text, want in ((TWO, 2), (THREE, 3), (MIXED, 2)):
            for step in (1, 7, 10_000):
                with self.subTest(calls=want, step=step):
                    evs = parser_events(text, step, tools=[TOOL, {"name": "apply_patch",
                                                                  "parameters": {"properties": {"input": {"type": "string"}}}}])
                    pieces: dict[str, list[str]] = {}
                    for e in evs:
                        if e.kind == "tool_args":
                            pieces.setdefault(e.call.id, []).append(e.text)
                    done = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(done), want)
                    self.assertEqual(len(pieces), want)              # no piece carries an unknown call id
                    for c in done:
                        self.assertEqual(json.loads("".join(pieces[c.id])), c.arguments)

    def test_parse_tool_call_mints_a_distinct_id_for_the_same_body(self):
        body = "<function=write>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nsame\n</parameter>\n</function>"
        ids = CallIds()
        made = [parse_tool_call(body, {"name": "write", "parameters": {"properties": PROPS}}, ids) for _ in range(4)]
        self.assertEqual(len({c.id for c in made}), 4)
        self.assertEqual({json.dumps(c.arguments) for c in made}, {'{"path": "a.txt", "content": "same"}'})
        self.assertTrue(all(ID_RE.fullmatch(c.id) for c in made))

    def test_an_unfinished_call_keeps_the_id_it_was_announced_with(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = parser_events(CUT, step)
                started = [e.call for e in evs if e.kind == "tool_start"]
                done = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual(len(started), 2)
                self.assertEqual(len(done), 1)                       # the cut one was never completed
                self.assertNotEqual(started[0].id, started[1].id)
                self.assertEqual(done[0].id, started[0].id)          # it was not given the other call's id
                pieces = {}
                for e in evs:
                    if e.kind == "tool_args":
                        pieces.setdefault(e.call.id, []).append(e.text)
                self.assertEqual(json.loads("".join(pieces[started[0].id])), {"path": "a.txt", "content": "whole"})
                self.assertEqual("".join(pieces[started[1].id]),
                                 '{"path":"b.txt","content":"hal')   # not closed, and its own id throughout


# ------------------------------------------------------------------ the registry
class Registry(unittest.TestCase):
    def test_an_id_is_issued_once_even_against_the_conversations_own(self):
        ids = CallIds(["call_a", "call_b"])
        fresh = {ids.new() for _ in range(60)}
        self.assertEqual(len(fresh), 60)
        self.assertFalse(fresh & {"call_a", "call_b"})
        self.assertTrue(all(ID_RE.fullmatch(i) for i in fresh))
        items = {ids.new("fc") for _ in range(20)} | {ids.new("ctc") for _ in range(20)}
        self.assertEqual(len(items), 40)
        self.assertFalse(items & fresh)
        self.assertTrue(all(i.startswith(("fc_", "ctc_")) for i in items))

    def test_note_takes_one_id_or_a_list(self):
        ids = CallIds()
        ids.note("call_x")
        ids.note(["call_y", None, ""])
        self.assertEqual(ids.used, {"call_x", "call_y"})
        fresh = ids.new()
        self.assertNotIn(fresh, {"call_x", "call_y"})
        self.assertIn(fresh, ids.used)              # an id is recorded before it is handed out


# ------------------------------------------------------------------ the Responses layer
def responses_stream(text: str, step: int = 1, tools=None, ids=None, include=False):
    """Drive OutputParser and the Assembler over `text` and return (the events, the parser's calls)."""
    tl, names, _ = request_tools({"tools": tools or RESP_TOOLS})
    ids = ids if ids is not None else CallIds()
    asm = Assembler({"model": "m", "input": "go"}, "m", 0, names, include, ids=ids)
    p = OutputParser(thinking=True, tools=tl, stream_tools=True, ids=ids)
    events = list(asm.start())
    parsed = []
    for i in range(0, len(text), step):
        for ev in p.feed(text[i:i + step]):
            parsed.append(ev)
            events += asm.feed(ev)
    parsed += p.finish()
    events += asm.finish({"finish": "stop", "completion_tokens": 40, "prompt_tokens": 7, "reused": 0,
                          "reasoning_tokens": 4})
    return events, [e.call for e in parsed if e.kind == "tool_call"]


class ResponsesEvents(unittest.TestCase):
    def check(self, events, want_calls):
        """The invariant, stated once, checked on any Responses event stream."""
        self.assertEqual(events[-1]["type"], "response.completed")
        output = events[-1]["response"]["output"]
        calls = [o for o in output if o["type"] in ("function_call", "custom_tool_call")]
        self.assertEqual(len(calls), want_calls)
        # 1. uniqueness within the response
        self.assertEqual(len({c["call_id"] for c in calls}), want_calls)
        self.assertEqual(len({o["id"] for o in output}), len(output))
        self.assertTrue(all(c["call_id"] and c["call_id"].isascii() for c in calls))
        # 2. one id per call across its whole life, and a call announced exactly once
        for c in calls:
            added = [e for e in events if e["type"] == "response.output_item.added"
                     and e.get("item", {}).get("call_id") == c["call_id"]]
            self.assertEqual(len(added), 1, c["call_id"])
            self.assertEqual(added[0]["item"]["id"], c["id"])
            self.assertEqual(added[0]["item"]["type"], c["type"])
            mine = [e for e in events if e["type"] in CALL_EVENTS and e.get("item_id") == c["id"]]
            self.assertTrue(mine)
            self.assertTrue(all(e.get("item", {}).get("call_id", c["call_id"]) == c["call_id"] for e in mine),
                            [e["type"] for e in mine])
            done = [e for e in events if e["type"] == "response.output_item.done" and e["item"]["id"] == c["id"]]
            self.assertEqual(len(done), 1)
            self.assertEqual(done[0]["item"], c)
        # 3. every argument piece reaches the item of the call it belongs to, and the whole is their concatenation
        pieces: dict[str, list[str]] = {}
        for e in events:
            if e["type"] == "response.function_call_arguments.delta":
                pieces.setdefault(e["item_id"], []).append(e["delta"])
            elif e["type"] == "response.custom_tool_call_input.delta":
                pieces.setdefault(e["item_id"], []).append(e["delta"])
        for c in calls:
            whole = c["arguments"] if c["type"] == "function_call" else c["input"]
            self.assertEqual("".join(pieces.get(c["id"], [])), whole)
            if c["type"] == "function_call":
                json.loads(whole)
        # 4. output_index on every item-bearing event, one value per item, equal to its place in the output
        used: dict[str, set[int]] = {}
        for e in events:
            if e["type"] not in ITEM_EVENTS:
                continue
            self.assertIn("output_index", e, e["type"])
            used.setdefault(e.get("item_id") or e["item"]["id"], set()).add(e["output_index"])
        self.assertEqual(set(used), {o["id"] for o in output})
        for at, o in enumerate(output):
            self.assertEqual(used[o["id"]], {at}, o["type"])

    def test_two_and_three_calls_and_a_custom_tool(self):
        for text, want in ((TWO, 2), (THREE, 3)):
            for step in (1, 7, 10_000):
                with self.subTest(calls=want, step=step):
                    events, _ = responses_stream(text, step)
                    self.check(events, want)

    def test_a_function_call_and_a_custom_tool_call(self):
        events, _ = responses_stream(MIXED, 1, tools=CUSTOM_TOOLS)
        self.check(events, 2)
        fc, ctc = [o for o in events[-1]["response"]["output"] if o["type"] == "function_call"][0], \
            [o for o in events[-1]["response"]["output"] if o["type"] == "custom_tool_call"][0]
        self.assertEqual((fc["name"], json.loads(fc["arguments"])), ("write", {"path": "a.txt", "content": "text"}))
        self.assertEqual((ctc["name"], ctc["input"]), ("apply_patch", "*** Begin Patch\n"))

    def test_an_unfinished_call_is_its_own_incomplete_item(self):
        events, _ = responses_stream(CUT, 1)
        output = events[-1]["response"]["output"]
        calls = [o for o in output if o["type"] == "function_call"]
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["call_id"], calls[1]["call_id"])
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual([c["status"] for c in calls], ["completed", "incomplete"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "a.txt", "content": "whole"})
        self.assertEqual(calls[1]["arguments"], '{"path":"b.txt","content":"hal')
        self.assertEqual([e["output_index"] for e in events if e["type"] == "response.output_item.added"],
                         list(range(len(output))))

    def test_streamed_events_and_the_completed_object_are_the_same_calls(self):
        """#5: the non-streaming answer is the `response` of the last event of the same stream."""
        events, _ = responses_stream(THREE, 1)
        whole = collect(iter(events))
        calls = [o for o in whole["output"] if o["type"] == "function_call"]
        self.assertEqual(len({c["call_id"] for c in calls}), 3)
        self.assertEqual([c["call_id"] for c in calls],
                         [e["item"]["call_id"] for e in events if e["type"] == "response.output_item.added"
                          and e["item"]["type"] == "function_call"])
        self.assertEqual([c["id"] for c in calls],
                         [e["item"]["id"] for e in events if e["type"] == "response.output_item.added"
                          and e["item"]["type"] == "function_call"])

    def test_the_ids_of_a_replayed_turn_are_never_reissued(self):
        """#7: a new turn of the same conversation cannot reuse an id an earlier turn already used."""
        first, calls = responses_stream(THREE, 1)
        used = {o["call_id"] for o in first[-1]["response"]["output"] if o["type"] == "function_call"}
        used |= {o["id"] for o in first[-1]["response"]["output"]}
        # 3 call ids + one item id per output item (3 function_call items and the reasoning item of "Thinking.")
        self.assertEqual(len(used), len(calls) + len(first[-1]["response"]["output"]))
        ids = CallIds()
        input_messages({"input": [{"role": "user", "content": "go"},
                                  *first[-1]["response"]["output"],
                                  *[{"type": "function_call_output", "call_id": c.id, "output": "ok"}
                                    for c in calls]]}, ids)
        again, _ = responses_stream(THREE, 1, ids=ids)
        newer = {o["call_id"] for o in again[-1]["response"]["output"] if o["type"] == "function_call"}
        newer |= {o["id"] for o in again[-1]["response"]["output"]}
        self.assertFalse(newer & used, newer & used)

    def test_a_result_naming_an_unknown_call_id_is_refused(self):
        items = [{"role": "user", "content": "go"},
                 {"type": "function_call", "call_id": "call_a", "name": "write", "arguments": '{"path":"a"}'},
                 {"type": "function_call_output", "call_id": "call_zzz", "output": "no"}]
        with self.assertRaises(ResponsesError) as e:
            input_messages({"input": items})
        self.assertEqual(e.exception.param, "input[2].call_id")
        self.assertIn("call_zzz", str(e.exception))

    def test_results_are_put_back_in_the_order_of_the_calls_they_answer(self):
        msgs = input_messages({"input": [{"role": "user", "content": "go"},
                                         {"type": "function_call", "call_id": "call_a", "name": "write",
                                          "arguments": '{"path":"1"}'},
                                         {"type": "function_call", "call_id": "call_b", "name": "write",
                                          "arguments": '{"path":"2"}'},
                                         {"type": "function_call_output", "call_id": "call_b", "output": "second"},
                                         {"type": "function_call_output", "call_id": "call_a", "output": "first"}]})
        self.assertEqual([m["content"] for m in msgs if m["role"] == "tool"], ["first", "second"])

    def test_a_result_with_no_call_id_is_still_tolerated(self):
        msgs = input_messages({"input": [{"role": "user", "content": "go"},
                                         {"type": "function_call", "call_id": "call_a", "name": "write",
                                          "arguments": "{}"},
                                         {"type": "function_call_output", "output": "no id given"}]})
        self.assertEqual([m["content"] for m in msgs if m["role"] == "tool"], ["no id given"])


# ------------------------------------------------------------------ the Chat Completions layer
def chat_stream(script: str, tools=None, call_ids=None):
    tok, svc = service(script)
    tools = tools or [TOOL]
    ids, thinking, max_new = svc.prepare([{"role": "user", "content": "go"}], tools, {}, 0)
    cancel = threading.Event()
    with contextlib.redirect_stdout(io.StringIO()):
        chunks = list(openai_chunks(svc, {"model": "m"}, ids, thinking, tools, max_new, cancel,
                                    call_ids=call_ids if call_ids is not None else CallIds()))
    return chunks


def chat_view(chunks):
    """What a streaming client builds: per call index, its id, its name, the arguments it was sent."""
    by: dict[int, dict] = {}
    for c in chunks:
        if c is None:
            continue
        for tc in c["choices"][0]["delta"].get("tool_calls") or []:
            cur = by.setdefault(tc.get("index", len(by)), {"id": None, "name": "", "arguments": ""})
            cur["id"] = tc.get("id") or cur["id"]
            cur["name"] += (tc.get("function") or {}).get("name") or ""
            cur["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    return [by[i] for i in sorted(by)]


class ChatLayer(unittest.TestCase):
    def test_parallel_calls_get_distinct_ids_and_their_own_arguments(self):
        for script, want in ((TWO, 2), (THREE, 3)):
            with self.subTest(calls=want):
                got = chat_view(chat_stream(script))
                self.assertEqual(len(got), want)
                self.assertEqual(len({c["id"] for c in got}), want)
                self.assertTrue(all(ID_RE.fullmatch(c["id"]) for c in got))
                self.assertEqual((got[0]["name"], json.loads(got[0]["arguments"])),
                                 (got[1]["name"], json.loads(got[1]["arguments"])))
                self.assertEqual(json.loads(got[0]["arguments"]), {"path": "a.txt", "content": "same"})

    def test_streamed_chunks_and_the_aggregated_answer_agree(self):
        """#5: openai_collect is the same chunks merged, so the ids and arguments are the same calls."""
        chunks = chat_stream(THREE)
        streamed = chat_view(chunks)
        msg = openai_collect(iter(chunks))["choices"][0]["message"]
        whole = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                 for c in msg["tool_calls"]]
        self.assertEqual(streamed, whole)
        self.assertEqual(len({c["id"] for c in whole}), 3)
        self.assertEqual(openai_collect(iter(chunks))["choices"][0]["finish_reason"], "tool_calls")

    def test_an_unfinished_call_keeps_its_own_id(self):
        chunks = chat_stream(CUT)
        got = chat_view(chunks)
        self.assertEqual(len(got), 2)
        self.assertNotEqual(got[0]["id"], got[1]["id"])
        self.assertEqual(json.loads(got[0]["arguments"]), {"path": "a.txt", "content": "whole"})
        self.assertEqual(got[1]["arguments"], '{"path":"b.txt","content":"hal')
        # the whole answer leaves the cut call out, and does not report "tool_calls" (#211)
        msg = openai_collect(iter(chunks))["choices"][0]
        self.assertEqual(msg["finish_reason"], "stop")
        self.assertEqual([c["id"] for c in msg["message"]["tool_calls"]], [got[0]["id"]])

    def test_the_registry_is_seeded_with_the_ids_the_client_used(self):
        ids = CallIds()
        openai_to_messages({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "write", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": "ok"}]}, ids)
        self.assertTrue({"call_a"} <= ids.used)
        fresh = ids.new()                                      # ... and a new one is not one of them
        self.assertEqual(ids.used, {"call_a", fresh})

    def test_a_result_naming_an_unknown_call_id_is_refused(self):
        with self.assertRaises(ToolCallPairingError) as e:
            openai_to_messages({"messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "write", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_zzz", "content": "no"}]})
        self.assertEqual(e.exception.param, "messages[2].tool_call_id")
        self.assertIn("call_zzz", str(e.exception))

    def test_results_are_put_back_in_the_order_of_the_calls_they_answer(self):
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "write", "arguments": '{"path":"1"}'}},
                {"id": "call_b", "type": "function", "function": {"name": "write", "arguments": '{"path":"2"}'}}]},
            {"role": "tool", "tool_call_id": "call_b", "content": "second"},
            {"role": "tool", "tool_call_id": "call_a", "content": "first"},
            {"role": "user", "content": "and now?"}]})
        self.assertEqual([m["content"] for m in msgs if m["role"] == "tool"], ["first", "second"])
        self.assertEqual(msgs[1]["tool_calls"][0], {"function": {"name": "write", "arguments": {"path": "1"}}})

    def test_pair_tool_results_leaves_the_messages_as_they_were(self):
        """The private ids are dropped, so the template's messages (and the existing tests' exact dicts) are unchanged."""
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": "ok"}]})
        self.assertEqual(msgs[1]["tool_calls"], [{"function": {"name": "f", "arguments": {"x": 1}}}])
        self.assertEqual(msgs[2], {"role": "tool", "content": "ok"})
        pair_tool_results(msgs)                                    # idempotent on already-clean messages


# ------------------------------------------------------------------ the Anthropic layer
def anthropic_stream(script: str, tools=None, call_ids=None):
    tok, svc = service(script)
    tools = tools or [TOOL]
    ids, thinking, max_new = svc.prepare([{"role": "user", "content": "go"}], tools, {}, 0)
    cancel = threading.Event()
    with contextlib.redirect_stdout(io.StringIO()):
        return list(anthropic_events(svc, {"model": "m"}, ids, thinking, tools, max_new, cancel,
                                     call_ids=call_ids if call_ids is not None else CallIds()))


def anthropic_view(events):
    """What a streaming client builds: per block, its tool_use id, name, and the input pieces it was sent."""
    blocks: dict[int, dict] = {}
    for name, e in events:
        if name == "content_block_start" and e["content_block"]["type"] == "tool_use":
            blocks[e["index"]] = {"id": e["content_block"]["id"], "name": e["content_block"]["name"], "json": ""}
        elif name == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
            blocks[e["index"]]["json"] += e["delta"]["partial_json"]
    return [blocks[i] for i in sorted(blocks)]


class AnthropicLayer(unittest.TestCase):
    def test_parallel_calls_get_distinct_ids_and_their_own_input(self):
        for script, want in ((TWO, 2), (THREE, 3)):
            with self.subTest(calls=want):
                got = anthropic_view(anthropic_stream(script))
                self.assertEqual(len(got), want)
                self.assertEqual(len({c["id"] for c in got}), want)
                self.assertTrue(all(ID_RE.fullmatch(c["id"]) for c in got))
                self.assertEqual(json.loads(got[0]["json"]), json.loads(got[1]["json"]))
                self.assertEqual(json.loads(got[0]["json"]), {"path": "a.txt", "content": "same"})

    def test_streamed_blocks_and_the_aggregated_message_agree(self):
        events = anthropic_stream(THREE)
        streamed = anthropic_view(events)
        msg = anthropic_collect(iter(events))
        whole = [{"id": b["id"], "name": b["name"], "json": json.dumps(b["input"], ensure_ascii=False)}
                 for b in msg["content"] if b["type"] == "tool_use"]
        self.assertEqual([(c["id"], c["name"]) for c in streamed], [(c["id"], c["name"]) for c in whole])
        for a, b in zip(streamed, whole):
            self.assertEqual(json.loads(a["json"]), json.loads(b["json"]))
        self.assertEqual(msg["stop_reason"], "tool_use")

    def test_an_unfinished_call_keeps_its_own_id(self):
        events = anthropic_stream(CUT)
        got = anthropic_view(events)
        self.assertEqual(len(got), 2)
        self.assertNotEqual(got[0]["id"], got[1]["id"])
        self.assertEqual(got[1]["json"], '{"path":"b.txt","content":"hal')
        msg = anthropic_collect(iter(events))
        self.assertEqual(msg["stop_reason"], "end_turn")            # not "tool_use": it cannot be run
        self.assertEqual([b["id"] for b in msg["content"] if b["type"] == "tool_use"], [got[0]["id"]])

    def test_a_result_naming_an_unknown_tool_use_id_is_refused(self):
        with self.assertRaises(ToolCallPairingError) as e:
            anthropic_to_messages({"messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_a", "name": "write", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_zzz", "content": "no"}]}]},
                True)
        self.assertEqual(e.exception.param, "messages[2].content.tool_use_id")
        self.assertIn("call_zzz", str(e.exception))

    def test_results_are_put_back_in_the_order_of_the_calls_they_answer(self):
        msgs, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_a", "name": "write",
                                               "input": {"path": "1"}},
                                              {"type": "tool_use", "id": "call_b", "name": "write",
                                               "input": {"path": "2"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_b", "content": "second"},
                                         {"type": "tool_result", "tool_use_id": "call_a", "content": "first"}]}]}, True)
        self.assertEqual([m["content"] for m in msgs if m["role"] == "tool"], ["first", "second"])


# ------------------------------------------------------------------ over HTTP
class OverHttp(unittest.TestCase):
    def setUp(self):
        self.tok, self.svc = service([TWO, THREE, ANSWER])
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def replay(self, *scripts):
        """Make the mock engine answer the next requests with these scripts (it replays encoded token ids)."""
        end = self.tok.encode(IM_END, parse_special=True)
        self.svc.engine.scripts = [self.tok.encode(s, parse_special=True) + end for s in scripts]
        self.svc.engine.script, self.svc.engine.turns = self.svc.engine.scripts[0], 0

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                                   **(headers or {})})
                r = c.getresponse()
                raw = r.read().decode()
            if r.getheader("Content-Type") == "text/event-stream":
                events = []
                for block in raw.split("\n\n"):
                    data = [l[len("data: "):] for l in block.splitlines() if l.startswith("data: ")]
                    if not data or not data[0].startswith("{"):
                        continue                      # a keep-alive comment, or the stream's [DONE]
                    events.append(json.loads(data[0]))
                return r.status, events
            return r.status, json.loads(raw)
        finally:
            c.close()

    def test_two_identical_calls_over_all_three_apis(self):
        self.replay(TWO)
        code, r = self.post("/v1/chat/completions", {"model": "m", "stream": False, "tools": OPENAI_TOOLS,
                                                     "messages": [{"role": "user", "content": "save both"}]})
        self.assertEqual(code, 200, r)
        calls = r["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(len({c["id"] for c in calls}), 2)
        self.assertEqual(calls[0]["function"]["arguments"], calls[1]["function"]["arguments"])

        code, r = self.post("/v1/messages", {"model": "m", "max_tokens": 500, "tools": ANTHROPIC_TOOLS,
                                            "messages": [{"role": "user", "content": "save both"}]})
        self.assertEqual(code, 200, r)
        use = [b for b in r["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(use), 2)
        self.assertEqual(len({b["id"] for b in use}), 2)
        self.assertEqual(use[0]["input"], use[1]["input"])

        code, r = self.post("/v1/responses", {"model": "m", "input": "save both", "tools": RESP_TOOLS})
        self.assertEqual(code, 200, r)
        fc = [o for o in r["output"] if o["type"] == "function_call"]
        self.assertEqual(len(fc), 2)
        self.assertEqual(len({c["call_id"] for c in fc}), 2)
        self.assertEqual(len({c["id"] for c in fc}), 2)
        self.assertEqual(fc[0]["arguments"], fc[1]["arguments"])

    def test_a_streamed_chat_answer_carries_both_ids_from_the_start(self):
        self.replay(TWO)
        code, chunks = self.post("/v1/chat/completions", {"model": "m", "stream": True, "tools": OPENAI_TOOLS,
                                                          "messages": [{"role": "user", "content": "save both"}]})
        self.assertEqual(code, 200)
        got = chat_view(chunks)
        self.assertEqual(len(got), 2)
        self.assertEqual(len({c["id"] for c in got}), 2)
        self.assertEqual(json.loads(got[0]["arguments"]), json.loads(got[1]["arguments"]))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_a_streamed_responses_answer_keeps_each_calls_ids_and_index(self):
        """#2 and #4 over the wire: the ids of response.output_item.added are those of the item's later events, of
        response.output_item.done and of the item inside response.completed, and one output_index per item."""
        self.replay(THREE)
        code, events = self.post("/v1/responses", {"model": "m", "stream": True, "tools": RESP_TOOLS,
                                                   "input": "save all three"})
        self.assertEqual(code, 200)
        added = [e for e in events if e["type"] == "response.output_item.added"
                 and e["item"]["type"] == "function_call"]
        self.assertEqual(len(added), 3)
        done = {e["item"]["id"]: e["item"] for e in events if e["type"] == "response.output_item.done"}
        whole = events[-1]["response"]["output"]
        calls = [o for o in whole if o["type"] == "function_call"]
        self.assertEqual(len({c["call_id"] for c in calls}), 3)
        for e in added:
            item = e["item"]
            mine = [x for x in events if x["type"] in CALL_EVENTS and x.get("item_id") == item["id"]]
            self.assertTrue(mine)
            self.assertTrue(all(x["output_index"] == e["output_index"] for x in mine),
                            [(x["type"], x["output_index"]) for x in mine])
            self.assertEqual(done[item["id"]]["call_id"], item["call_id"])
            self.assertEqual(done[item["id"]]["arguments"],
                             "".join(x["delta"] for x in mine
                                     if x["type"] == "response.function_call_arguments.delta"))
        self.assertEqual([o["item"]["id"] for o in added], [c["id"] for c in calls])
        self.assertEqual([o["item"]["call_id"] for o in added], [c["call_id"] for c in calls])
        # one output_index per item, on every event that names it
        used: dict[str, set] = {}
        for e in events:
            if e["type"] in ITEM_EVENTS:
                used.setdefault(e.get("item_id") or e["item"]["id"], set()).add(e["output_index"])
        self.assertEqual(used, {o["id"]: {i} for i, o in enumerate(whole)})

    def test_a_replayed_turn_pairs_every_result_with_its_own_call(self):
        code, r = self.post("/v1/responses", {"model": "m", "input": "save both", "tools": RESP_TOOLS})
        fc = [o for o in r["output"] if o["type"] == "function_call"]
        self.assertEqual(len(fc), 2)
        # the client sends the results back in the order they finished, not the order of the calls
        code, r2 = self.post("/v1/responses", {"model": "m", "tools": RESP_TOOLS, "input": [
            {"type": "message", "role": "user", "content": "save both"}, *r["output"],
            {"type": "function_call_output", "call_id": fc[1]["call_id"], "output": "second-result"},
            {"type": "function_call_output", "call_id": fc[0]["call_id"], "output": "first-result"}]})
        self.assertEqual(code, 200, r2)
        prompt = self.tok.decode(self.svc.engine.last_prompt)
        # the model sees the results in the order of the calls, not the order they finished in (the words are
        # hyphenated so they cannot be found in the template's own text)
        self.assertLess(prompt.index("first-result"), prompt.index("second-result"))
        # both calls of the replayed turn are in the prompt with the arguments they were sent with
        self.assertEqual(prompt.count("<parameter=path>"), 2)
        self.assertEqual(prompt.count("a.txt"), 2)
        # and the ids of this turn are new ones, so a late result cannot attach to the wrong call
        newer = {o["call_id"] for o in r2["output"] if o["type"] == "function_call"}
        self.assertFalse(newer & {c["call_id"] for c in fc})

    def test_a_result_naming_an_unknown_call_id_is_a_400(self):
        base = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "write", "arguments": "{}"}}]}]
        code, r = self.post("/v1/chat/completions", {"model": "m", "messages": [
            *base, {"role": "tool", "tool_call_id": "call_zzz", "content": "no"}]})
        self.assertEqual(code, 400, r)
        self.assertEqual(r["error"]["type"], "invalid_request_error")
        self.assertIn("call_zzz", r["error"]["message"])

        code, r = self.post("/v1/messages", {"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_a", "name": "write", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_zzz", "content": "no"}]}]})
        self.assertEqual(code, 400, r)
        self.assertIn("call_zzz", r["error"]["message"])

        code, r = self.post("/v1/responses", {"model": "m", "input": [
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "call_a", "name": "write", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_zzz", "output": "no"}]})
        self.assertEqual(code, 400, r)
        self.assertEqual(r["error"]["param"], "input[2].call_id")
        self.assertIn("call_zzz", r["error"]["message"])

    def test_a_turn_with_no_ids_at_all_still_works(self):
        """Older clients (and this server's own MCP rounds) send tool results with no id: nothing here refuses them."""
        code, r = self.post("/v1/chat/completions", {"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "write", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": "and now?"}]})
        self.assertEqual(code, 200, r)


if __name__ == "__main__":
    unittest.main()
