"""Actual model tokenizer count only. Never import GPU libraries or read weights."""
import argparse
import json
import sys
import pathlib
from xtml_codec_r143 import KimiXTMLCodec


def cases():
    def tool(name, properties, required):
        return {"type": "function", "function": {"name": name,
                "parameters": {"type": "object", "properties": properties, "required": required}}}
    tools = [tool("read_file", {"path": {"type": "string"}}, ["path"]),
             tool("calculate", {"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]),
             tool("write_file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])]
    messages = [{"role": "system", "content": "Use tools. Never invent tool results."},
                {"role": "user", "content": "Read input.json. Add its a and b with calculate. Write nonce and sum to out.json, read it, report both."}]
    output = [{"label": "initial", "messages": list(messages), "tools": tools}]
    steps = [("read_file", {"path": "input.json"}, {"nonce": "P143A9", "a": 137, "b": 269}),
             ("calculate", {"a": 137, "b": 269}, {"sum": 406}),
             ("write_file", {"path": "out.json", "content": '{"nonce":"P143A9","sum":406}'}, {"written": True}),
             ("read_file", {"path": "out.json"}, {"nonce": "P143A9", "sum": 406})]
    for i, (name, arguments, result) in enumerate(steps, 1):
        call_id = "c%d" % i
        messages += [{"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": call_id, "name": name,
             "content": json.dumps(result, separators=(",", ":"))}]
        output.append({"label": "after_tool_%d" % i, "messages": list(messages), "tools": tools if i < 4 else None})
    return output


def main():
    p = argparse.ArgumentParser();p.add_argument("--model", default="/shared/Kimi-K3");p.add_argument("--output");args=p.parse_args()
    codec = KimiXTMLCodec(args.model)
    results = []
    for case in cases():
        text, ids = codec.encode_chat(case["messages"], {"tools": case["tools"], "thinking": False})
        results.append({"label": case["label"], "prompt_tokens": len(ids),
                        "output_budget_if_1024": 1024-len(ids), "fits_historical_512_prompt_gate": len(ids)<=512,
                        "fits_1024_with_128_output": len(ids)+128<=1024,
                        "rendered_characters": len(text), "truncated": False})
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    result={"CPU_only": True, "GPU": False, "weights_read": False,
                      "tokenizer": codec.identity, "purpose": "SIMULATED_HISTORY_TOKEN_COUNT_NOT_MODEL_OR_TOOL_ACCEPTANCE",
                      "results": results, "all_fit_512": all(x["fits_historical_512_prompt_gate"] for x in results),
                      "all_fit_1024_plus128": all(x["fits_1024_with_128_output"] for x in results)}
    if args.output:
        with pathlib.Path(args.output).open("x",encoding="utf8") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__ == "__main__":main()
