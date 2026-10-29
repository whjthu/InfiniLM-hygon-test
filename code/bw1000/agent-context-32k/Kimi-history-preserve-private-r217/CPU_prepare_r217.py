"""Stdlib-only execution of actual segment renderers; no BPE or Model claim."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL_SHA = "49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676"
CHECKS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS[name] = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def payload(segments):
    return [(s.text, s.allow_special) for s in segments]


def text(segments):
    return "".join(s.text for s in segments)


def run():
    old_path = ROOT / "baseline/frozen_encoding_k3_r143.py"
    new_path = ROOT / "runtime/frozen_encoding_k3_r143.py"
    check("baseline_exact_r205_pin", sha(old_path) == ORIGINAL_SHA)
    check("generic_codec_byte_unchanged", sha(ROOT / "runtime/xtml_codec_r143.py") == "7618b171dd40d58548d6707e40fdb88c1666fda8bd23d7107d6e8f7ad070a4d7")
    original_long = (ROOT / "baseline/long_codec_r179.py").read_text(encoding="utf-8")
    candidate_long = (ROOT / "runtime/long_codec_r179.py").read_text(encoding="utf-8")
    check("long_codec_only_one_explicit_optin_added", candidate_long.replace("  kwargs['preserve_history_reasoning']=True\n", "", 1) == original_long and candidate_long.count("  kwargs['preserve_history_reasoning']=True\n") == 1)
    old = load("r217_baseline_renderer", old_path)
    new = load("frozen_encoding_k3_r143", new_path)
    old_tree = ast.parse(old_path.read_text(encoding="utf-8"))
    new_tree = ast.parse(new_path.read_text(encoding="utf-8"))
    old_func = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_chat_segments")
    new_func = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_chat_segments")
    index = next(i for i, n in enumerate(new_func.args.kwonlyargs) if n.arg == "preserve_history_reasoning")
    check("new_flag_default_false", isinstance(new_func.args.kw_defaults[index], ast.Constant) and new_func.args.kw_defaults[index].value is False)
    # Existing generation prefix and effort branches must remain AST-identical.
    for key in ("add_generation_prompt", "thinking"):
        a = [n for n in old_func.body if isinstance(n, ast.If) and (isinstance(n.test, ast.Name) and n.test.id == key or key == "thinking" and "thinking_effort" in ast.dump(n.test))]
        b = [n for n in new_func.body if isinstance(n, ast.If) and (isinstance(n.test, ast.Name) and n.test.id == key or key == "thinking" and "thinking_effort" in ast.dump(n.test))]
        check("unchanged_generation_branch_" + key, [ast.dump(n) for n in a] == [ast.dump(n) for n in b])
    old_funcs = {n.name: ast.dump(n) for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: ast.dump(n) for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    check("all_other_renderer_helpers_AST_identical", all(new_funcs[name] == value for name, value in old_funcs.items() if name != "build_chat_segments"))

    history = [{"role": "system", "content": "fixture system"}]
    markers = []
    ordinary = "ordinary <|open|>think<|sep|> [EOS] must stay text"
    for i in range(8):
        marker = "r217_history_%d <|close|>response<|sep|>" % i
        markers.append(marker)
        history.extend([
            {"role": "user", "content": "step %d " % i + ordinary},
            {"role": "assistant", "content": "answer %d" % i,
             "reasoning_content" if i % 2 == 0 else "reasoning": marker,
             "tool_calls": [{"id": "call_%d" % i, "type": "function", "function": {"name": "calculate", "arguments": "{\"x\":1,\"y\":2}"}}]},
            {"role": "tool", "tool_call_id": "call_%d" % i, "content": "result %d " % i + ordinary},
        ])
    history.append({"role": "user", "content": "continue"})
    before = copy.deepcopy(history)
    op = "<|open|>"
    sep = "<|sep|>"
    cl = "<|close|>"
    for thinking in (False, True):
        options = {"thinking": thinking, "thinking_effort": "high"}
        baseline = old.build_chat_segments(history, **options)
        default = new.build_chat_segments(history, **options)
        explicit_false = new.build_chat_segments(history, preserve_history_reasoning=False, **options)
        preserved = new.build_chat_segments(history, preserve_history_reasoning=True, **options)
        check("default_segment_byte_equivalence_%s" % thinking, payload(default) == payload(baseline) == payload(explicit_false))
        rendered = text(preserved)
        for i, marker in enumerate(markers):
            check("history_reasoning_%d_once_mode_%s" % (i, thinking), rendered.count(marker) == 1)
            check("history_reasoning_%d_in_think_mode_%s" % (i, thinking), (op + "think" + sep + marker + cl + "think" + sep) in rendered)
            check("history_reasoning_%d_ordinary_segment_mode_%s" % (i, thinking), any(s.text == marker and not s.allow_special for s in preserved))
        check("generation_prefix_mode_%s" % thinking, rendered.endswith(op + ("think" if thinking else "response") + sep))
        check("effort_depends_only_current_mode_%s" % thinking, ("thinking-effort" in rendered) == thinking)
        check("structural_looking_ordinary_stays_ordinary_%s" % thinking, all(not s.allow_special for s in preserved if ordinary in s.text))
        check("tool_call_and_result_retained_%s" % thinking, rendered.count("calculate") == 16 and all(rendered.count("answer %d" % i) == 1 and rendered.count("result %d " % i) == 1 for i in range(8)))
        if thinking:
            check("thinking_mode_unchanged_with_optin", payload(preserved) == payload(baseline))
    check("caller_eight_step_history_unchanged", history == before)

    no_rc = [{"role": "user", "content": ordinary}, {"role": "assistant", "content": ordinary}, {"role": "user", "content": "next"}]
    for thinking in (False, True):
        for prefix in (False, True):
            for options in ({}, {"thinking_effort": "high"}, {"response_format": {"type": "json_object"}, "tool_choice": "none"}):
                kwargs = dict(options, thinking=thinking, add_generation_prompt=prefix)
                a = old.build_chat_segments(no_rc, **kwargs)
                b = new.build_chat_segments(no_rc, preserve_history_reasoning=True, **kwargs)
                check("no_RC_equivalence_%s_%s_%s" % (thinking, prefix, len(CHECKS)), payload(a) == payload(b))
    # Whitespace is explicitly valid reasoning in the original helper; empty is not.
    for value, present in (("", False), (" ", True), (None, False)):
        h = [{"role": "assistant", "reasoning_content": value, "content": "answer"}]
        b = new.build_chat_segments(h, thinking=False, preserve_history_reasoning=True, add_generation_prompt=False)
        check("empty_whitespace_semantics_%r" % value, (op + "think" + sep in text(b)) == present)
    for module in (old, new):
        try:
            module.build_chat_segments([{"role": "developer", "content": "existing behavior"}])
        except ValueError:
            pass
        else:
            raise AssertionError("developer unexpectedly accepted")
    check("developer_remains_rejected_no_new_transform", True)

    # Exercise the real long codec method and real renderer with a stdlib segment
    # recorder. This checks routing/allow-special only, never claims BPE tokens.
    sys.path.insert(0, str(ROOT / "runtime"))
    long = load("r217_long_codec", ROOT / "runtime/long_codec_r179.py")
    class SegmentRecorder:
        def __init__(self):
            self.calls = []
        def encode(self, value, **kwargs):
            self.calls.append((value, kwargs))
            return []
    codec = object.__new__(long.KimiLongCodec)
    codec.encoding = SegmentRecorder()
    kwargs = {"enable_thinking": False, "preserve_history_reasoning": False}
    original_kwargs = copy.deepcopy(kwargs)
    rendered, ids = codec.encode_chat(history, kwargs)
    check("long_codec_forces_scoped_optin", all(rendered.count(marker) == 1 for marker in markers))
    check("long_codec_does_not_force_current_thinking", rendered.endswith(op + "response" + sep) and "thinking-effort" not in rendered)
    check("long_codec_options_and_messages_unchanged", kwargs == original_kwargs and history == before)
    check("codec_ordinary_text_uses_disallowed_special_empty", all(call_kwargs == {"disallowed_special": ()} for value, call_kwargs in codec.encoding.calls if ordinary in value or value in markers))
    base = object.__new__(long.KimiXTMLCodec)
    base.encoding = SegmentRecorder()
    generic_rendered, _ = base.encode_chat(history, {"enable_thinking": False})
    check("generic_codec_default_still_drops_RC_in_chat", all(marker not in generic_rendered for marker in markers))
    for options, effort in (({"enable_thinking": True}, "high"), ({"enable_thinking": True, "reasoning_effort": "low"}, "low")):
        actual, _ = codec.encode_chat(history, options)
        check("long_codec_current_thinking_effort_" + effort, actual.endswith(op + "think" + sep) and ("thinking_effort=" + effort) in actual)
    forbidden = [name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "transformers" or name.startswith("transformers.") or name == "tiktoken"]
    check("no_Torch_Transformers_tiktoken_import", not forbidden)
    return {"status": "PASSED", "CPU_only": True, "actual_renderer_executed": True,
            "checks": CHECKS, "check_count": len(CHECKS),
            "historical_reasoning_fields": 8, "BPE_executed": False, "Native_or_Model_executed": False,
            "developer_transform": "NOT_APPLICABLE: actual renderer rejects developer; no transform added",
            "scope": "segment renderer and codec option routing only; actual BPE/saved-history/model qualification remain unproven",
            "source_SHA256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in (old_path, new_path, ROOT / "runtime/xtml_codec_r143.py", ROOT / "runtime/long_codec_r179.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": result["check_count"], "output_SHA256": sha(args.output)}))
