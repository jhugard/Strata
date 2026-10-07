"""Offline tests of tools/tool_call_ids_http_smoke.py: the checks it runs against a live server must also fail
when a stream breaks the invariant.  No GPU, no server, no network - canned event streams, each one mutated in the
way that would mis-pair a client's bookkeeping.

    python -m unittest tools.test_tool_call_ids_http_smoke
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.tool_call_ids_http_smoke import (anthropic_stream_checks, chat_stream_checks,  # noqa: E402
                                            events_of, responses_stream_checks, run, whole_checks)

A, B = "call_" + "a" * 24, "call_" + "b" * 24
IA, IB = "fc_" + "1" * 32, "fc_" + "2" * 32
ARGS_A, ARGS_B = '{"path":"a.txt"}', '{"path":"b.txt"}'


def responses_events(call_ids=(A, B), item_ids=(IA, IB), args=(ARGS_A, ARGS_B), *, dup_id=False,
                     drop_index=None, move_index=None, cross=False, wrong_name=False, drop_done=False,
                     completed=True, extra_delta=None):
    """A two-call Responses stream, mutated as asked."""
    ids = list(zip(call_ids, item_ids, args))
    if dup_id:
        ids[1] = (ids[0][0], ids[1][1], ids[1][2])
    events = [{"type": "response.created", "response": {"id": "rs_1"}},
              {"type": "response.output_item.added", "output_index": 0,
               "item": {"id": "rs_msg", "type": "message", "content": []}},
              {"type": "response.output_text.delta", "item_id": "rs_msg", "output_index": 0, "delta": "Reading."},
              {"type": "response.output_item.done", "output_index": 0,
               "item": {"id": "rs_msg", "type": "message", "content": [{"type": "output_text", "text": "Reading."}]}}]
    for n, (cid, iid, arg) in enumerate(ids):
        at = n + 1
        events.append({"type": "response.output_item.added", "output_index": at,
                       "item": {"id": iid, "type": "function_call", "call_id": cid, "name": "read", "arguments": ""}})
        pieces = [arg[:len(arg) // 2], arg[len(arg) // 2:]] if len(arg) > 1 else [arg]
        for piece in pieces:
            events.append({"type": "response.function_call_arguments.delta",
                           "item_id": ids[1 - n][1] if cross else iid, "output_index": at, "delta": piece})
        if extra_delta is not None:
            events.append({"type": "response.function_call_arguments.delta", "item_id": extra_delta,
                           "output_index": at, "delta": "x"})
        events.append({"type": "response.function_call_arguments.done", "item_id": iid, "output_index": at,
                       "name": "other" if wrong_name else "read", "arguments": arg})
        if drop_done:
            continue
        events.append({"type": "response.output_item.done", "output_index": at,
                       "item": {"id": iid, "type": "function_call", "call_id": cid, "name": "read", "arguments": arg}})
    for e in events:
        if drop_index is not None and e.get("item_id") == drop_index and e["type"].endswith(".delta"):
            e.pop("output_index")
        if move_index is not None and e.get("item_id") == move_index:
            e["output_index"] = 7
    if completed:
        output = [{"id": iid, "type": "function_call", "call_id": cid, "name": "read", "arguments": arg}
                  for cid, iid, arg in ids]
        events.append({"type": "response.completed", "response": {"id": "rs_1", "status": "completed",
                                                                  "output": output}})
    return events


class ResponsesStream(unittest.TestCase):
    def test_a_correct_stream_passes(self):
        calls, problems = responses_stream_checks(responses_events())
        self.assertEqual(problems, [])
        self.assertEqual([c["call_id"] for c in calls], [A, B])
        self.assertEqual([c["arguments"] for c in calls], [ARGS_A, ARGS_B])

    def test_two_calls_sharing_a_call_id_is_reported(self):
        _, problems = responses_stream_checks(responses_events(dup_id=True))
        self.assertTrue(any("share a call_id" in p for p in problems), problems)

    def test_two_calls_sharing_an_item_id_is_reported(self):
        _, problems = responses_stream_checks(responses_events(item_ids=(IA, IA)))
        self.assertTrue(any("share an item id" in p for p in problems), problems)

    def test_a_dsh_join_key_that_is_not_unique_is_reported(self):
        """DSH keys everything on `${call_id}|${id}`; a collision there is what makes a session unrestorable."""
        _, problems = responses_stream_checks(responses_events(dup_id=True, item_ids=(IA, IA)))
        self.assertTrue(any("`${call_id}|${id}`" in p for p in problems), problems)

    def test_an_argument_piece_on_another_calls_item_is_reported(self):
        _, problems = responses_stream_checks(responses_events(cross=True))
        self.assertTrue(any("its deltas are" in p for p in problems), problems)

    def test_a_piece_on_an_item_that_was_never_announced_is_reported(self):
        _, problems = responses_stream_checks(responses_events(extra_delta="fc_" + "9" * 32))
        self.assertTrue(any("never announced" in p for p in problems), problems)

    def test_a_missing_output_index_is_reported(self):
        _, problems = responses_stream_checks(responses_events(drop_index=IA))
        self.assertTrue(any("no output_index" in p for p in problems), problems)

    def test_an_item_that_moves_between_output_indices_is_reported(self):
        _, problems = responses_stream_checks(responses_events(move_index=IB))
        self.assertTrue(any("moves between output indices" in p for p in problems), problems)

    def test_two_items_sharing_an_output_index_is_reported(self):
        events = responses_events()
        for e in events:
            if e.get("item_id") == IB or (e.get("item") or {}).get("id") == IB:
                e["output_index"] = 1
        _, problems = responses_stream_checks(events)
        self.assertTrue(any("share an output_index" in p for p in problems), problems)

    def test_completed_disagreeing_with_the_announcements_is_reported(self):
        events = responses_events()
        events[-1]["response"]["output"][0]["call_id"] = "call_" + "z" * 24
        _, problems = responses_stream_checks(events)
        self.assertTrue(any("different call_ids" in p for p in problems), problems)

    def test_a_call_announced_twice_is_reported(self):
        events = responses_events()
        at = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.added"
                  and e["item"]["type"] == "function_call")
        events.insert(at, dict(events[at]))
        _, problems = responses_stream_checks(events)
        self.assertTrue(any("announced" in p and "closed" in p for p in problems), problems)

    def test_a_name_that_changes_mid_stream_is_reported(self):
        _, problems = responses_stream_checks(responses_events(wrong_name=True))
        self.assertTrue(any("closed as" in p for p in problems), problems)

    def test_an_id_that_is_not_opaque_is_reported(self):
        _, problems = responses_stream_checks(responses_events(call_ids=("call_read_1", B)))
        self.assertTrue(any("opaque" in p for p in problems), problems)

    def test_a_stream_without_response_completed_is_reported(self):
        calls, problems = responses_stream_checks(responses_events(completed=False))
        self.assertEqual(calls, [])
        self.assertTrue(any("response.completed" in p for p in problems), problems)


class ChatStream(unittest.TestCase):
    @staticmethod
    def chunks(ids=(A, B), *, dup=False, shared_index=False, orphan=False):
        out = [{"choices": [{"delta": {"role": "assistant"}}]}]
        for n, (cid, arg) in enumerate(zip(ids, (ARGS_A, ARGS_B))):
            index = 0 if shared_index else n
            out.append({"choices": [{"delta": {"tool_calls": [{"index": index, "id": cid,
                                                               "function": {"name": "read"}}]}}]})
            out.append({"choices": [{"delta": {"tool_calls": [{"index": index,
                                                               "function": {"arguments": arg}}]}}]})
        if dup:
            out.append({"choices": [{"delta": {"tool_calls": [{"index": 2, "id": ids[0],
                                                               "function": {"name": "read"}}]}}]})
        if orphan:
            out.append({"choices": [{"delta": {"tool_calls": [{"index": 5,
                                                               "function": {"arguments": "x"}}]}}]})
        out.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        return out

    def test_a_correct_stream_passes(self):
        calls, problems = chat_stream_checks(self.chunks())
        self.assertEqual(problems, [])
        self.assertEqual([c["id"] for c in calls], [A, B])
        self.assertEqual([c["arguments"] for c in calls], [ARGS_A, ARGS_B])

    def test_an_id_announced_twice_is_reported(self):
        _, problems = chat_stream_checks(self.chunks(dup=True))
        self.assertTrue(any("announced twice" in p for p in problems), problems)

    def test_two_calls_on_one_index_is_reported(self):
        _, problems = chat_stream_checks(self.chunks(shared_index=True))
        self.assertTrue(any("carries two calls" in p for p in problems), problems)

    def test_arguments_on_an_index_no_call_opened_is_reported(self):
        _, problems = chat_stream_checks(self.chunks(orphan=True))
        self.assertTrue(any("which no call opened" in p for p in problems), problems)

    def test_two_calls_sharing_an_id_is_reported(self):
        _, problems = chat_stream_checks(self.chunks(ids=(A, A)))
        self.assertTrue(any("share an id" in p for p in problems), problems)


class AnthropicStream(unittest.TestCase):
    @staticmethod
    def events(ids=(A, B), *, dup=False, orphan=False):
        out = [{"type": "message_start", "message": {"id": "msg_1"}},
               {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
               {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Reading."}},
               {"type": "content_block_stop", "index": 0}]
        for n, (cid, arg) in enumerate(zip(ids, (ARGS_A, ARGS_B))):
            out.append({"type": "content_block_start", "index": n + 1,
                        "content_block": {"type": "tool_use", "id": cid, "name": "read"}})
            out.append({"type": "content_block_delta", "index": n + 1,
                        "delta": {"type": "input_json_delta", "partial_json": arg}})
            out.append({"type": "content_block_stop", "index": n + 1})
        if dup:
            out.append({"type": "content_block_start", "index": 9,
                        "content_block": {"type": "tool_use", "id": ids[0], "name": "read"}})
        if orphan:
            out.append({"type": "content_block_delta", "index": 8,
                        "delta": {"type": "input_json_delta", "partial_json": "x"}})
        out.append({"type": "message_stop"})
        return out

    def test_a_correct_stream_passes(self):
        calls, problems = anthropic_stream_checks(self.events())
        self.assertEqual(problems, [])
        self.assertEqual([c["id"] for c in calls], [A, B])

    def test_a_block_id_opened_twice_is_reported(self):
        _, problems = anthropic_stream_checks(self.events(dup=True))
        self.assertTrue(any("opened as two blocks" in p for p in problems), problems)

    def test_input_for_a_block_that_was_never_opened_is_reported(self):
        _, problems = anthropic_stream_checks(self.events(orphan=True))
        self.assertTrue(any("never opened" in p for p in problems), problems)

    def test_two_blocks_sharing_an_id_is_reported(self):
        _, problems = anthropic_stream_checks(self.events(ids=(A, A)))
        self.assertTrue(any("share an id" in p for p in problems), problems)


class WholeAgainstStream(unittest.TestCase):
    def test_the_same_calls_agree(self):
        stream = [{"name": "read", "arguments": ARGS_A}, {"name": "read", "arguments": ARGS_B}]
        whole = [{"id": A, "function": {"name": "read", "arguments": ARGS_A}},
                 {"id": B, "function": {"name": "read", "arguments": ARGS_B}}]
        self.assertEqual(whole_checks(whole, stream), [])

    def test_a_different_call_set_is_reported(self):
        stream = [{"name": "read", "arguments": ARGS_A}, {"name": "read", "arguments": ARGS_B}]
        whole = [{"id": A, "function": {"name": "read", "arguments": ARGS_A}}]
        self.assertTrue(any("the whole answer describes" in p for p in whole_checks(whole, stream)))

    def test_a_reused_id_in_the_whole_answer_is_reported(self):
        stream = [{"name": "read", "arguments": ARGS_A}, {"name": "read", "arguments": ARGS_B}]
        whole = [{"id": A, "function": {"name": "read", "arguments": ARGS_A}},
                 {"id": A, "function": {"name": "read", "arguments": ARGS_B}}]
        self.assertTrue(any("reuses an id" in p for p in whole_checks(whole, stream)))


THREE_CALLS = [{"id": IA, "type": "function_call", "call_id": A, "name": "read", "arguments": ARGS_A},
               {"id": IB, "type": "function_call", "call_id": B, "name": "read", "arguments": ARGS_B},
               {"id": "fc_" + "3" * 32, "type": "function_call", "call_id": "call_" + "c" * 24, "name": "read",
                "arguments": '{"path":"c.txt"}'}]


def http_error(body):
    return urllib.error.HTTPError("http://127.0.0.1:8080/v1/responses", 400, "Bad Request", {},
                                  io.BytesIO(json.dumps(body).encode()))


class FakeClient:
    """Answers like the server does when the invariant holds, from the same canned streams."""

    def __init__(self):
        self.model = "fake"

    def calls_for(self, prompt, extra_input):
        if extra_input:
            return [o for o in extra_input if o.get("type") == "function_call"]
        if prompt.startswith("Read a.txt, b.txt and c.txt"):
            return THREE_CALLS
        if prompt.startswith("Call read twice"):              # the identical-arguments prompt
            return [dict(THREE_CALLS[0]), dict(THREE_CALLS[1], id=IB, call_id=B, arguments=ARGS_A)]
        return THREE_CALLS[:2]

    def responses(self, prompt, stream=False, extra_input=()):
        calls = self.calls_for(prompt, extra_input)
        known = {c["call_id"] for c in calls}
        for o in extra_input:
            if o.get("type") == "function_call_output" and o["call_id"] not in known:
                raise http_error({"error": {"message": f"a tool result for call_id {o['call_id']!r}, which no tool "
                                                        "call of the assistant turn before it issued",
                                            "type": "invalid_request_error", "param": "input[4].call_id"}})
        if stream:
            return 200, responses_events([c["call_id"] for c in calls], [c["id"] for c in calls],
                                         [c["arguments"] for c in calls])
        outs = [o for o in extra_input if o.get("type") == "function_call_output"]
        if outs:
            # Pair each result with the call its call_id names, in the order of the calls, the way
            # serve.frontend.pair_tool_results does.
            by_id = {o["call_id"]: o["output"] for o in outs}
            text = "\n".join(f'{json.loads(c["arguments"])["path"]} {by_id[c["call_id"]]}' for c in calls)
            return 200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        return 200, {"output": calls}

    def chat(self, prompt, stream=False):
        if stream:
            return 200, ChatStream.chunks()
        return 200, {"choices": [{"message": {"tool_calls": [{"id": A, "function": {"name": "read",
                                                                                     "arguments": ARGS_A}},
                                                                     {"id": B, "function": {"name": "read",
                                                                                           "arguments": ARGS_B}}]},
                                    "finish_reason": "tool_calls"}]}

    def anthropic(self, prompt, stream=False):
        if stream:
            return 200, AnthropicStream.events()
        return 200, {"content": [{"type": "tool_use", "id": A, "name": "read", "input": {"path": "a.txt"}},
                                 {"type": "tool_use", "id": B, "name": "read", "input": {"path": "b.txt"}}]}


class Run(unittest.TestCase):
    def test_run_reports_every_check(self):
        result = run(FakeClient())
        names = [c["name"] for c in result["checks"]]
        for want in ("responses stream: parallel reads", "responses whole: the same calls",
                     "chat stream: parallel reads", "anthropic stream: parallel reads", "three parallel calls",
                     "replay: each result reached the call its call_id named",
                     "an unknown call_id is refused with the field in param"):
            self.assertIn(want, names)
        self.assertTrue(all(c["ok"] for c in result["checks"]), result["checks"])

    def test_a_broken_stream_makes_run_fail(self):
        class Broken(FakeClient):
            def responses(self, prompt, stream=False, extra_input=()):
                code, events = super().responses(prompt, stream, extra_input)
                if stream and "both files" in prompt:
                    added = [e for e in events if e["type"] == "response.output_item.added"
                             and e["item"]["type"] == "function_call"]
                    added[1]["item"]["call_id"] = added[0]["item"]["call_id"]        # two calls, one id
                return code, events

        result = run(Broken())
        bad = [c["name"] for c in result["checks"] if not c["ok"]]
        self.assertIn("responses stream: parallel reads", bad)


class DryRun(unittest.TestCase):
    def test_without_run_nothing_is_sent(self):
        out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]
                                                  / "tools/tool_call_ids_http_smoke.py")],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Dry run", out.stdout)
        self.assertNotIn("PASS:", out.stdout)


class EventsOf(unittest.TestCase):
    def test_only_json_data_lines_are_kept(self):
        raw = ('data: {"type": "a"}\n\n: keep-alive\n\ndata: [DONE]\n\ndata: {"type": "b"}\n\n')
        self.assertEqual([e["type"] for e in events_of(raw)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
