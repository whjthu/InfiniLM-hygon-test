"""Bounded CPU real-BPE audit. Never reads model outputs or prints private data."""
import argparse
import copy
import hashlib
import importlib
import json
import pathlib
import subprocess
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path("/refactor/incoming/Kimi-agent-service-Model-r205")
MODEL = pathlib.Path("/shared/Kimi-K3")
TOKENIZER = {"tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
             "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"}
CANDIDATE = {"frozen_encoding_k3_r143.py": "5baf623402041af20ea074186134eef2db4f5f1ef4708133805a2566a734149f",
             "long_codec_r179.py": "01a4a921ab27f1ce78514b0ab4994060c9c95b2bdf362e354b877b8fd06c4143",
             "xtml_codec_r143.py": "7618b171dd40d58548d6707e40fdb88c1666fda8bd23d7107d6e8f7ad070a4d7"}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def require(value):
    if not value:
        raise AssertionError("bounded CPU BPE contract failed")


def safe_error(error):
    return {"error_class": type(error).__name__, "error_SHA256": hashlib.sha256(str(error).encode()).hexdigest()}


def fixture():
    # Eight synthetic closed tool steps. Never represented as Model output.
    ordinary = "ordinary [EOS] <|open|>think<|sep|> <|close|>response<|sep|>"
    messages = [{"role": "system", "content": "Synthetic CPU history fixture only."}]
    markers = []
    for i in range(8):
        marker = "synthetic history reasoning %d " % i + ordinary
        markers.append(marker)
        messages.extend([
            {"role": "user", "content": "step %d " % i + ordinary},
            {"role": "assistant", "content": "answer %d" % i,
             "reasoning_content" if i % 2 == 0 else "reasoning": marker,
             "tool_calls": [{"id": "synthetic_%d" % i, "type": "function", "function": {"name": "calculate", "arguments": "{\"x\":1,\"y\":2}"}}]},
            {"role": "tool", "tool_call_id": "synthetic_%d" % i, "content": "result %d " % i + ordinary},
        ])
    messages.append({"role": "user", "content": "continue with the complete history"})
    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"], "additionalProperties": False}}}]
    return messages, tools, markers, ordinary


def arm(name):
    require(sha(ROOT / "replay_saved477_pins-r205.json") == "c26eba93283b63749c8a946d882e4368c9da2a7a3c3d73cd691a91b0f143abb6")
    pins = json.loads((ROOT / "replay_saved477_pins-r205.json").read_bytes())
    # Exact pure protocol import pattern from replay_saved477_r205.py. Do not
    # execute infinilm.__init__, load Native, or read the node generated_ids.
    sources = {}
    for rel, expected in pins["source_SHA256"].items():
        require(sha(BASE / rel) == expected)
        sources["baseline/" + rel] = expected
    require(sha(pins["saved_request"]) == pins["saved_request_SHA256"])
    runtime = BASE / "runtime" if name == "baseline" else ROOT / "candidate_runtime"
    if name == "candidate":
        for rel, expected in CANDIDATE.items():
            require(sha(runtime / rel) == expected)
            sources["candidate_runtime/" + rel] = expected
    for rel, expected in TOKENIZER.items():
        require(sha(MODEL / rel) == expected)
    for module_name, folder in (("infinilm", BASE / "overlay/python/infinilm"), ("infinilm.server", BASE / "overlay/python/infinilm/server")):
        require(module_name not in sys.modules)
        module = types.ModuleType(module_name)
        module.__path__ = [str(folder)]
        sys.modules[module_name] = module
    protocol = importlib.import_module("infinilm.server.agent_protocol")
    require(pathlib.Path(protocol.__file__).resolve() == BASE / "overlay/python/infinilm/server/agent_protocol.py")
    sys.path.insert(0, str(runtime))
    codec_module = importlib.import_module("long_codec_r179")
    renderer = importlib.import_module("frozen_encoding_k3_r143")
    generic_module = importlib.import_module("xtml_codec_r143")
    for mod, filename in ((codec_module, "long_codec_r179.py"), (renderer, "frozen_encoding_k3_r143.py"), (generic_module, "xtml_codec_r143.py")):
        require(pathlib.Path(mod.__file__).resolve() == runtime / filename)
    codec = codec_module.KimiLongCodec(str(MODEL))
    generic = generic_module.KimiXTMLCodec(str(MODEL))
    require(all(codec.identity[key] == expected for key, expected in (("tokenizer_config_SHA256", TOKENIZER["tokenizer_config.json"]), ("tiktoken_model_SHA256", TOKENIZER["tiktoken.model"]))))
    body = json.loads(pathlib.Path(pins["saved_request"]).read_bytes())
    before = copy.deepcopy(body)
    require(len(body["messages"]) == 4 and len(body["tools"]) == 3 and body["max_tokens"] == 4096 and body["parallel_tool_calls"] is False and not body["stream"])
    prepared, _, _ = protocol.prepare_agent_request(body, "kimi_k3")
    has_rc = any(m.get("role") == "assistant" and (m.get("reasoning_content") or m.get("reasoning")) not in (None, "") for m in prepared["messages"])
    require(not has_rc)
    prompt, ids = codec.encode_chat(prepared["messages"], prepared.get("chat_template_kwargs") or {}, bool(prepared.get("add_generation_prompt", True)))
    require(len(ids) == 8506 and all(type(i) is int and 0 <= i < 163840 for i in ids))
    require(prepared["max_tokens"] == 4096 and len(ids) + prepared["max_tokens"] <= 32768 and body == before)
    actual = {"input_tokens": len(ids), "output_budget": prepared["max_tokens"], "total_tokens": len(ids) + prepared["max_tokens"],
              "prompt_SHA256": hashlib.sha256(prompt.encode()).hexdigest(), "IDs_SHA256": digest_ids(ids),
              "full_request_schema_prepared": True, "caller_unchanged": True, "nonempty_historical_RC": has_rc}

    messages, tools, markers, ordinary = fixture()
    original = copy.deepcopy(messages)
    rows = {}
    for thinking in (False, True):
        budget = 8192 if thinking else 4096
        request = {"messages": messages, "tools": tools, "max_tokens": budget, "parallel_tool_calls": False,
                   "chat_template_kwargs": {"enable_thinking": thinking}}
        ready, _, _ = protocol.prepare_agent_request(request, "kimi_k3")
        options = ready["chat_template_kwargs"]
        rendered, encoded = codec.encode_chat(ready["messages"], options)
        empty = copy.deepcopy(ready["messages"])
        for message in empty:
            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
        _, without_rc = codec.encode_chat(empty, options)
        effective = dict(options)
        effective.pop("enable_thinking", None)
        effective["thinking"] = thinking
        if thinking:
            effective.setdefault("thinking_effort", "high")
        if name == "candidate":
            effective["preserve_history_reasoning"] = True
        segments = renderer.build_chat_segments(ready["messages"], **effective)
        require(rendered == "".join(s.text for s in segments))
        rc_present = all(rendered.count(marker) == 1 for marker in markers)
        expected_present = thinking or name == "candidate"
        require(rc_present == expected_present)
        ordinary_ids = codec.encoding.encode(ordinary, disallowed_special=())
        rc_contribution = sum(len(codec.encoding.encode(marker, disallowed_special=())) for marker in markers)
        require(all(i < 163584 for i in ordinary_ids))
        require(all(not segment.allow_special for segment in segments if ordinary in segment.text))
        if expected_present:
            require(all(any(s.text == marker and not s.allow_special for s in segments) for marker in markers))
            require(len(encoded) - len(without_rc) >= rc_contribution > 0)
        else:
            require(encoded == without_rc)
        require(rendered.endswith("<|open|>" + ("think" if thinking else "response") + "<|sep|>"))
        require(("thinking-effort" in rendered) == thinking)
        require(ready["max_tokens"] == budget and len(encoded) + budget <= 32768)
        legacy_text, legacy_ids = generic.encode_chat(ready["messages"], options)
        rows["thinking" if thinking else "chat"] = {
            "input_tokens": len(encoded), "without_RC_input_tokens": len(without_rc), "RC_text_tokens": rc_contribution,
            "output_budget": budget, "total_tokens": len(encoded) + budget, "budget_not_clamped": True,
            "historical_RC_preserved": rc_present, "current_generation_prefix_correct": True,
            "effort_depends_only_current_mode": True, "ordinary_special_looking_text_has_no_special_IDs": True,
            "prompt_SHA256": hashlib.sha256(rendered.encode()).hexdigest(), "IDs_SHA256": digest_ids(encoded),
            "legacy_prompt_SHA256": hashlib.sha256(legacy_text.encode()).hexdigest(), "legacy_IDs_SHA256": digest_ids(legacy_ids),
            "legacy_input_tokens": len(legacy_ids)}
    require(messages == original)
    require(not any(k == "torch" or k.startswith("torch.") or k == "transformers" or k.startswith("transformers.") for k in sys.modules))
    return {"status": "PASSED", "arm": name, "source_SHA256": sources, "tokenizer_SHA256": TOKENIZER,
            "A_saved_request": actual, "B_synthetic_eight_RC": rows, "module_paths_verified": True,
            "caller_fixture_unchanged": True, "CPU_only": True, "Native_imported": False,
            "real_BPE_executed": True, "model_output_report_read": False, "weights_read": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    args = parser.parse_args()
    if args.arm:
        try:
            result = arm(args.arm)
        except BaseException as error:
            result = {"status": "FAILED", "arm": args.arm, **safe_error(error)}
        print(json.dumps(result, separators=(",", ":")))
        return 0 if result["status"] == "PASSED" else 1
    require(args.output is not None)
    # x-mode reserves exactly one new result and never overwrites old evidence.
    with args.output.open("x", encoding="utf-8") as output:
        result = {"status": "FAILED", "CPU_only": True, "real_BPE_both_arms_completed": False, "GPU": False, "Model": False,
                  "synthetic_history_is_not_Model_evidence": True, "actual_saved_request_has_nonempty_RC": False,
                  "actual_history_RC_qualification": "NOT_COVERED", "isolated_subprocesses": True}
        try:
            results = {}
            for name in ("baseline", "candidate"):
                done = subprocess.run([sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "--arm", name], capture_output=True, text=True, timeout=240)
                child = json.loads(done.stdout)
                results[name] = child
                require(done.returncode == 0 and child["status"] == "PASSED")
            a, b = results["baseline"], results["candidate"]
            require(a["A_saved_request"] == b["A_saved_request"])
            for mode in ("chat", "thinking"):
                x, y = a["B_synthetic_eight_RC"][mode], b["B_synthetic_eight_RC"][mode]
                for key in ("legacy_prompt_SHA256", "legacy_IDs_SHA256", "legacy_input_tokens", "output_budget"):
                    require(x[key] == y[key])
                if mode == "thinking":
                    require(x["prompt_SHA256"] == y["prompt_SHA256"] and x["IDs_SHA256"] == y["IDs_SHA256"])
            result.update(status="PASSED", arms=results, real_BPE_both_arms_completed=True, A_full8506_prompt_and_IDs_equal=True,
                          B_candidate_preserves_eight_RC_both_modes=True, C_legacy_default_equivalent=True)
        except BaseException as error:
            result.update(safe_error(error))
            if "results" in locals():
                result["arms"] = results
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output_SHA256": sha(args.output), "CPU_only": True}))
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
