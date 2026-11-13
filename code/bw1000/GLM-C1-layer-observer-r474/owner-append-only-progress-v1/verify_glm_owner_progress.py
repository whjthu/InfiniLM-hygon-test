from pathlib import Path
import ast
import hashlib
import io
import json


OWNER = Path("code/bw1000/GLM-C1-layer-observer-r474/full-model-paired-worker-v1/owner.py")
ORIGINAL_SHA256 = "a6ff9551a60bf2853132668cc4b9da45499130407d1cb818911ffd1c44f04736"
CANDIDATE_SHA256 = "ada5b38631b55dfb9f68add839c0e67a7412d99aa1a5c87ae9863f1722fe1041"
PATCH_SHA256 = "89ddc8660969ebff8637d205b2c52190d738362fa356c292e3c693b0366dbef8"


def checked(raw, digest):
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Source or patch differs from the verified revision")
    return raw.decode("utf-8")


class Clock:
    def time(self):
        return 100.0

    def monotonic(self):
        return 50.0


def exercise(source):
    mark = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "mark")
    counts = dict(local=0, target=0)
    shared = []
    state = dict(phase=None, events=[], requests=[],
                 Native_forward_attempts=0, Native_forward_completed=0)
    trace = io.BytesIO()

    def save(path, value):
        counts[path] += 1
        if path == "target":
            shared.append(json.loads(json.dumps(value)))

    scope = dict(q=state, time=Clock(), json=json, trace=trace, save=save,
                 local="local", target="target")
    exec(compile(ast.Module(body=[mark], type_ignores=[]), "<owner mark>", "exec"), scope)
    mark = scope["mark"]
    mark("load.progress")
    request = dict(ordinal=179, generated_token_ids=[])
    state["requests"].append(request)
    mark("request.enter", ordinal=24)
    for position in range(4096):
        state["Native_forward_attempts"] += 1
        mark("forward.enter", kind="decode", tokens=1, past=position)
        state["Native_forward_completed"] += 1
        mark("forward.return", tokens=1, past=position, elapsed_s=0.3)
        request["generated_token_ids"].append(position)
    mark("request.return", ordinal=24, generated=4096)
    mark("destroy.return")
    return counts, [json.loads(line) for line in trace.getvalue().splitlines()], shared


def without_progress(event):
    return {key: value for key, value in event.items() if key != "progress"}


def main():
    original = checked(OWNER.read_bytes(), ORIGINAL_SHA256)
    patch = checked(Path(__file__).with_name("owner.patch").read_bytes(), PATCH_SHA256)
    hunk = patch.splitlines(keepends=True)[3:]
    old = "".join(line[1:] for line in hunk if line.startswith((" ", "-")))
    new = "".join(line[1:] for line in hunk if line.startswith((" ", "+")))
    assert original.count(old) == 1
    candidate = checked(original.replace(old, new, 1).encode(), CANDIDATE_SHA256)
    baseline, before, baseline_shared = exercise(original)
    updated, after, updated_shared = exercise(candidate)
    assert baseline == dict(local=8196, target=3)
    assert updated == dict(local=4, target=3)
    assert before == [without_progress(event) for event in after]
    for position in range(4096):
        enter, leave = after[2 + 2 * position:4 + 2 * position]
        assert enter["progress"] == dict(ordinal=179, generated=position,
                                         forward_attempts=position + 1, forward_completed=position)
        assert leave["progress"] == dict(ordinal=179, generated=position,
                                         forward_attempts=position + 1, forward_completed=position + 1)
    for actual, expected in zip(updated_shared, baseline_shared, strict=True):
        actual["events"] = [without_progress(event) for event in actual["events"]]
        assert actual == expected
    print(json.dumps(dict(baseline=baseline, candidate=updated, trace_events=len(after))))


if __name__ == "__main__":
    main()
