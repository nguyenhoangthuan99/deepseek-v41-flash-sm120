#!/usr/bin/env python3
"""Check the deployed API without changing server configuration or running external tools."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def events(response):
    data = []
    event_type = "message"
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event_type, "\n".join(data)
            data, event_type = [], "message"
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line.startswith("event:"):
            event_type = line[6:].lstrip(" ")
    require(not data, "Incomplete final SSE event")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    report = {
        "started_unix": time.time(),
        "base_url": base,
        "checks": [],
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply_integers",
                "description": "Multiply two integers and return their product.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool_messages = [
        {
            "role": "system",
            "content": "For multiplication, call multiply_integers. Do not calculate it yourself. Once the tool returns, answer with its result and do not call it again.",
        },
        {
            "role": "user",
            "content": "Use multiply_integers to multiply 17 by 23. Wait for the tool result before giving the answer.",
        },
    ]

    def get(path):
        with urllib.request.urlopen(base + path, timeout=15) as response:
            return response.status, response.read()

    def request(label, *, stream=False, **fields):
        body = {
            "model": "deepseek-v41-flash",
            "temperature": 0,
            "max_tokens": 1024,
            "stream": stream,
            **fields,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        record = {"case": label, "request": body}
        report["checks"].append(record)
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            record["http_status"] = response.status
            if not stream:
                value = json.load(response)
                record["response"] = value
                require(not value.get("error"), f"{label}: API error")
                return value["choices"][0]
            text, reasoning, calls, raw_events = [], [], {}, []
            finish = usage = None
            done = False
            for event_type, payload in events(response):
                require(event_type != "error", payload)
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                raw_events.append(event)
                require(not event.get("error"), f"{label}: SSE error")
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
                    for part in delta.get("tool_calls") or []:
                        call = calls.setdefault(
                            part["index"],
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        call["id"] += part.get("id") or ""
                        function = part.get("function") or {}
                        call["function"]["name"] += function.get("name") or ""
                        call["function"]["arguments"] += function.get("arguments") or ""
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
            value = {
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": "".join(text),
                    "reasoning_content": "".join(reasoning),
                    "tool_calls": [calls[index] for index in sorted(calls)],
                },
            }
            record.update(events=raw_events, assembled=value, usage=usage, done=done)
            require(
                done and finish is not None and usage is not None,
                f"{label}: incomplete stream",
            )
            return value

    def check_answer(choice, thinking=False):
        message = choice["message"]
        require(choice["finish_reason"] == "stop", "Answer did not finish normally")
        require(
            "391" in (message.get("content") or ""), "Expected arithmetic answer 391"
        )
        require(
            bool(message.get("reasoning_content")) == thinking,
            "Unexpected reasoning field",
        )
        require(
            "<think>" not in message["content"]
            and "</think>" not in message["content"],
            "Raw thinking tags leaked into answer",
        )

    def check_tool(choice, thinking=False):
        message = choice["message"]
        require(
            choice["finish_reason"] == "tool_calls",
            "Expected structured tool-call finish",
        )
        calls = message.get("tool_calls") or []
        require(
            len(calls) == 1 and bool(calls[0].get("id")),
            "Expected one identified tool call",
        )
        call = calls[0]
        require(
            call["function"]["name"] == "multiply_integers", "Unexpected function name"
        )
        values = json.loads(call["function"]["arguments"])
        require(
            values == {"a": 17, "b": 23}
            and all(type(v) is int for v in values.values()),
            "Unexpected typed tool arguments",
        )
        require(
            "DSML" not in (message.get("content") or ""),
            "Raw tool tags leaked into content",
        )
        require(
            bool(message.get("reasoning_content")) == thinking,
            "Unexpected tool reasoning field",
        )
        return call, values

    def passed():
        report["checks"][-1]["status"] = "passed"
        print(report["checks"][-1]["case"] + ": passed", flush=True)

    try:
        status, _ = get("/health")
        require(status == 200, "Server is not healthy")
        _, payload = get("/get_server_info")
        info = json.loads(payload)
        report["server"] = {
            key: info.get(key)
            for key in (
                "served_model_name",
                "context_length",
                "tp_size",
                "ep_size",
                "moe_runner_backend",
                "speculative_algorithm",
                "speculative_dspark_block_size",
                "tool_call_parser",
                "reasoning_parser",
            )
        }
        require(
            info.get("tool_call_parser") == "deepseekv41"
            and info.get("reasoning_parser") == "deepseek-v41",
            "V4.1 parser defaults are not enabled",
        )
        messages = [
            {
                "role": "user",
                "content": "What is 17 times 23? Reply with the number only.",
            }
        ]
        for label, stream, thinking, extra in (
            ("plain-chat", False, False, {}),
            ("reasoning-high", False, True, {"reasoning_effort": "high"}),
            ("reasoning-stream", True, True, {"reasoning_effort": "high"}),
            ("reasoning-none", False, False, {"reasoning_effort": "none"}),
        ):
            check_answer(
                request(label, stream=stream, messages=messages, **extra), thinking
            )
            passed()
        for label, stream, thinking, choice in (
            ("tool-auto", False, False, "auto"),
            ("tool-stream", True, False, "auto"),
            ("reasoning-tool", False, True, "auto"),
            ("reasoning-tool-stream", True, True, "auto"),
            ("tool-required", False, False, "required"),
            (
                "tool-named",
                False,
                False,
                {"type": "function", "function": {"name": "multiply_integers"}},
            ),
        ):
            extra = {"reasoning_effort": "high"} if thinking else {}
            answer = request(
                label,
                stream=stream,
                messages=tool_messages,
                tools=tools,
                tool_choice=choice,
                **extra,
            )
            call, values = check_tool(answer, thinking)
            passed()
            if label == "tool-auto":
                # Whitelisted local arithmetic only, never arbitrary model code.
                result = values["a"] * values["b"]
                assistant = {
                    "role": "assistant",
                    "content": answer["message"].get("content"),
                    "tool_calls": answer["message"]["tool_calls"],
                }
                continuation = tool_messages + [
                    assistant,
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps({"result": result}),
                    },
                ]
                check_answer(
                    request(
                        "tool-roundtrip",
                        messages=continuation,
                        tools=tools,
                        tool_choice="auto",
                    )
                )
                passed()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        if isinstance(error, urllib.error.HTTPError):
            report["http_error_body"] = error.read().decode("utf-8", errors="replace")
        raise
    finally:
        report["finished_unix"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
