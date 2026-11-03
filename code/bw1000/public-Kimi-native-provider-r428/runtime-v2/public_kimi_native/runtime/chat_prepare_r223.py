"""Kimi-only API intent rendering; original history and strict parser stay intact."""
import copy
import json


def prepare(messages, data, kwargs):
    options = dict(kwargs)
    # The validated top-level API choice is authoritative for this turn.
    options.pop("tool_choice", None)
    choice = data.get("tool_choice", "auto")
    if choice is None:
        choice = "auto"
    named = isinstance(choice, dict)
    if named:
        choice = choice["function"]["name"]
    parallel = data.get("parallel_tool_calls", True)
    if type(parallel) is not bool:
        raise ValueError("parallel_tool_calls must be boolean")
    tools = options.get("tools") or []
    rules = []
    if choice == "none" and not named:
        # Reuse the frozen official renderer's explicit no-tools channel.
        options["tool_choice"] = "none"
    else:
        if named:
            if choice not in [tool["function"]["name"] for tool in tools]:
                raise ValueError("Requested tool unavailable after protocol filtering")
            options["tool_choice"] = "required"
            rules.append("This assistant turn requires a real call to function " + json.dumps(choice, ensure_ascii=False) + " following its provided schema. Do not replace the required call with a text answer.")
        elif choice == "required":
            options["tool_choice"] = "required"
        elif choice != "auto":
            raise ValueError("Unsupported tool choice")
        if tools and not parallel:
            rules.append("In this assistant turn, issue at most one real tool call. After that call, end this assistant turn and wait for its tool result before deciding or calling any next tool. Do not combine later steps or additional invocations in this turn.")
    # Never combine, remove or edit any historical message/content/RC/tool result.
    # Auto+paralleltrue adds nothing and retains the old options byte semantics.
    derived = copy.deepcopy(messages)
    if rules:
        derived.insert(len(derived) if options.get("thinking", True) else 0, {"role": "system", "content": "[Request tool-call constraints]\n" + "\n".join(rules)})
    return derived, options


def encode(codec, messages, data, add_generation_prompt, kwargs):
    derived, options = prepare(messages, data, kwargs)
    return codec.encode_chat(derived, options, add_generation_prompt)
