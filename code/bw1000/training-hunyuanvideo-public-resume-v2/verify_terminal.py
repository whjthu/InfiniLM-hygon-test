"""Verify original HunyuanVideo v2 sender and receiver terminal summaries."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def verify(arm, plan, sender, receiver, launch):
    intent = receiver["intent"]
    outcome = receiver["outcome"]
    require(isinstance(intent, dict) and isinstance(outcome, dict),
            "Receiver terminal receipt required")
    request = launch["request_id"]
    require(isinstance(request, str) and re.fullmatch(r"[0-9a-f]{32}", request),
            "Invalid request ID")
    require(launch["claim"] == "e124758359448cf6ff5d05ed4141783b7a745bf8",
            "Original publication differs")
    require(launch["deadline"] - launch["epoch"] == plan["seconds"],
            "Original launch budget differs")
    for receipt in (sender, intent, outcome, receiver["progress"], receiver["last_fsynced"]):
        if receipt is not None:
            require(all(receipt.get(key) == launch[key] for key in ("request_id", "claim", "deadline")),
                    "Receipt differs from original launch")
    total = plan["total"]
    offset = outcome["offset"]
    sent = sender["bytes_sent"]
    require(type(offset) is int and type(sent) is int, "Offsets must be integers")
    require(plan["offset"] <= offset <= total and 0 <= sent <= total - plan["offset"], "Invalid extent")
    durable = []
    for field in ("progress", "last_fsynced"):
        row = receiver[field]
        if row is not None:
            require(row["request_id"] == request, "Unrelated durable receipt")
            require(type(row["offset"]) is int and plan["offset"] <= row["offset"] <= offset, "Invalid durable extent")
            require(re.fullmatch(r"[0-9a-f]{64}", row["prefix_SHA256"]), "Invalid durable prefix hash")
            durable.append(row)
    if outcome["passed"] is True:
        require(sender["passed"] is True and sender["rc"] == 0 and sender["error"] is None, "Sender did not confirm success")
        require(offset == total == receiver["final_bytes"] and receiver["partial_bytes"] is None, "Final extent differs")
        require(sent == total - plan["offset"], "Sent extent differs")
        require(outcome["whole_SHA256"] == plan["full_SHA256"], "Whole checksum differs")
        sender_receipt = sender["receiver"]
        require(isinstance(sender_receipt, dict) and sender_receipt.get("passed") is True
                and sender_receipt.get("offset") == total and sender_receipt.get("request_id") == request
                and sender_receipt.get("whole_SHA256") == plan["full_SHA256"],
                "Sender observed a different final receipt")
        require(sender_receipt.get("claim") == launch["claim"]
                and sender_receipt.get("deadline") == launch["deadline"],
                "Sender's final receipt differs from original launch")
        return dict(asset=arm, request_id=request, state="SENDER_AND_RECEIVER_SUCCESS", bytes=total, receiver_whole_SHA256=plan["full_SHA256"])
    require(outcome["passed"] is False and sender["passed"] is False, "Contradictory terminal states")
    require(receiver["final_bytes"] is None and receiver["partial_bytes"] == offset, "Failed receiver extent differs")
    if len(durable) == 2 and durable[0]["offset"] == durable[1]["offset"]:
        require(durable[0]["prefix_SHA256"] == durable[1]["prefix_SHA256"],
                "Durable receipts disagree at the same offset")
    require(offset <= plan["offset"] + sent, "Receiver extent exceeds sender extent")
    checkpoint = max(durable, key=lambda row: row["offset"], default=None)
    return dict(asset=arm, request_id=request, state="TERMINAL_FAILURE", partial_bytes=offset,
                receiver_error=outcome["error_type"], sender_error=sender["error"],
                durable_offset=checkpoint["offset"] if checkpoint else plan["offset"],
                durable_prefix_SHA256=checkpoint["prefix_SHA256"] if checkpoint else plan["prefix_SHA256"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("arm", choices=("vae", "transformer"))
    parser.add_argument("launch_intent", type=Path)
    parser.add_argument("sender_terminal", type=Path)
    parser.add_argument("receiver_observation", type=Path)
    args = parser.parse_args()
    plan, plan_sha = read(args.plan)
    require(plan_sha == "d2d58e2514b54be97f56def4d85fced1153536064c111c6c06a64e37601d681b",
            "Frozen transfer plan differs")
    sender, sender_sha = read(args.sender_terminal)
    observed, observation_sha = read(args.receiver_observation)
    launch, launch_sha = read(args.launch_intent)
    result = verify(args.arm, plan["assets"][args.arm], sender, observed["assets"][args.arm], launch)
    print(json.dumps(dict(result, plan_SHA256=plan_sha, sender_SHA256=sender_sha,
                          receiver_observation_SHA256=observation_sha, launch_intent_SHA256=launch_sha), indent=2))


if __name__ == "__main__":
    main()
