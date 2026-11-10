"""Resolve the two preregistered original R450 mode-0 requests without exporting private inputs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(32 * 1024**2 + 1)
    require(0 < len(raw) <= 32 * 1024**2 and sha(raw) == expected,
            "pinned file absent, out of bounds, or different: " + path.name)
    return raw


def by_id(rows, key):
    index = {row[key]: row for row in rows}
    require(len(index) == len(rows) == 256, "original 256-case inventory differs")
    return index


def resolve(root, spec, *, host_shared=None):
    def from_pin(pin):
        path = PurePosixPath(pin["path"])
        require(path.is_absolute() and path.parts[1] == "refactor", "unexpected asset namespace")
        rel = Path(*path.parts[2:])
        return json.loads(pinned(root / rel, pin["SHA256"]))

    enc = from_pin(spec["encoding"])
    req = from_pin(spec["requests"])
    cases = from_pin(spec["cases"])["cases"]
    require(enc["completed"] is True and enc["passed"] is True
            and enc["actual_Deep_codec"] is True and enc["actual_public_protocol"] is True
            and enc["actual_encodings"] == 256 and enc["EOS_ids"] == [1],
            "original encoding qualification differs")
    lookup_cases = by_id(cases, "id")
    lookup_requests = by_id(req["requests"], "case_id")
    lookup_encoding = by_id(enc["cases"], "case_id")
    require(lookup_cases.keys() == lookup_requests.keys() == lookup_encoding.keys(),
            "case sets differ across pinned inputs")
    logical = req["logical_plan"]
    require(len(logical) == 512 and sha(compact(logical)) == spec["logical_plan_SHA256"],
            "original logical plan differs")
    selected = []
    for row in spec["selected"]:
        i = row["logical_index"]
        entry = {key: value for key, value in row.items() if key != "suite"}
        require(logical[i] == entry and i == 2 * row["case_ordinal"] and row["mode"] == 0,
                "selected position differs from original plan")
        case = lookup_cases[row["case_id"]]
        request = lookup_requests[row["case_id"]]
        encoding = lookup_encoding[row["case_id"]]
        body = request["body"]
        raw = compact(body)
        require(case["suite"] == encoding["suite"] == row["suite"]
                and encoding["selected_ordinal"] == row["case_ordinal"]
                and encoding["source_case_SHA256"] == sha(compact(case))
                and sha(raw) == request["request_body_SHA256"] == encoding["request_body_SHA256"]
                == row["request_body_SHA256"]
                and encoding["prompt_IDs_SHA256"] == row["prompt_IDs_SHA256"]
                and encoding["prompt_tokens"] == row["prompt_tokens"]
                and encoding["output_budget"] == row["output_budget"] == 4096
                and encoding["input_plus_output"] == row["prompt_tokens"] + 4096 <= 32768,
                "original case, exact request bytes, or encoded prompt differs")
        require(body["stream"] is True and body["text_delivery"] == "P1"
                and body["max_tokens"] == 4096 and body["temperature"] == 0
                and body["chat_template_kwargs"]["enable_thinking"] is False,
                "original P1 chat sampling differs")
        selected.append((row, raw, case))
    require([row["suite"] for row, _, _ in selected] == ["gsm8k", "mmlu"],
            "two first-case pilot slots required")
    processor = root / spec["processor_relative"]
    for name, digest in spec["processor_files_SHA256"].items():
        item = processor / name
        if host_shared is not None and item.is_symlink():
            link = PurePosixPath(item.readlink().as_posix())
            model = PurePosixPath("/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8")
            require(link == model / name, "processor link differs from the container mount contract")
            item = host_shared / Path(*link.parts[2:])
        pinned(item, digest)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha", required=True)
    args = parser.parse_args()
    spec = json.loads(pinned(args.spec, args.spec_sha))
    selected = resolve(args.root, spec)
    print(json.dumps({"kind": "i03_request_input_preflight", "input_binding_SHA256": args.spec_sha,
                      "selected": [row for row, _, _ in selected],
                      "processor_files_verified": len(spec["processor_files_SHA256"]),
                      "GPU": False, "HTTP": 0}, allow_nan=False))


if __name__ == "__main__":
    main()
