"""Read live Slurm and current per-card standby evidence for one exact request."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PREFIX = "training80-dcu-standby"
RECENT_SECONDS = 18


def document(path):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("standby metadata exceeded its bound")
    return json.loads(raw)


def accounting(label_prefix, parents):
    rows = {}
    for parent in parents:
        argv = ["sacct", "-n", "-P", "-j", parent,
                "-o", "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"]
        result = subprocess.check_output(argv, text=True, timeout=10)
        for line in result.splitlines():
            row = line.split("|")
            if len(row) == 7 and row[1].startswith(label_prefix):
                if row[1] in rows or not re.fullmatch(parent + r"\.[0-9]+", row[0]):
                    raise ValueError("standby accounting identity differs")
                rows[row[1]] = row
    return rows


def observe(request):
    if not re.fullmatch(r"[a-f0-9]{32}", request):
        raise ValueError("expected a full standby request")
    root = RESULTS / f"{PREFIX}-{request}"
    intent = document(root / "intent.json")
    receipt = document(root / "submission-receipt.json")
    if intent["request"] != request or receipt["request"] != request or intent["synthetic"] is not True:
        raise ValueError("standby receipt identity differs")
    records = receipt["submit_records"]
    rows = accounting(f"{PREFIX}-{request[:12]}-", sorted({record["parent"] for record in records}))
    now = time.time()
    details = {}
    busy_values = []
    for record in records:
        node, label = record["node"], record["label"]
        if node in details or not re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", node):
            raise ValueError("standby submission node is not unique")
        row = rows.get(label)
        status = dict(step=row[0] if row else None, state=row[2] if row else "UNOBSERVED",
                      actual_cards=0, recent_computing_cards=0)
        details[node] = status
        if row is None or row[2] != "RUNNING":
            continue
        if row[5] != node or "gres/dcu=8" not in row[6].split(","):
            raise ValueError("observed standby resource scope differs")
        status["actual_cards"] = 8
        directory = root / node
        try:
            worker = document(directory / "worker.json")
        except FileNotFoundError:
            continue
        if worker["step"] != row[0] or len(set(worker["pci"])) != 8:
            raise ValueError("standby worker step or device inventory differs")
        for index, pci in enumerate(worker["pci"]):
            try:
                pulse = document(directory / f"card-{index}.json")
            except FileNotFoundError:
                continue
            if (pulse["pci"] == pci and pulse["active"] is True and type(pulse["calls"]) is int and pulse["calls"] > 0
                    and -1 <= now - pulse["epoch"] <= RECENT_SECONDS):
                status["recent_computing_cards"] += 1
                busy = pulse["busy"]
                if type(busy) is int and 0 <= busy <= 100:
                    busy_values.append(busy)
    return dict(request=request, synthetic=True, observed_epoch=now, step_states=Counter(info["state"] for info in details.values()),
                actual_allocated_standby_cards=sum(info["actual_cards"] for info in details.values()),
                recent_computing_standby_cards=sum(info["recent_computing_cards"] for info in details.values()),
                asynchronous_samples=len(busy_values),
                asynchronous_busy_mean=sum(busy_values) / len(busy_values) if busy_values else None, nodes=details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    args = parser.parse_args()
    print(json.dumps(observe(args.request), sort_keys=True))
