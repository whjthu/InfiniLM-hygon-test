"""Submit the frozen answer-format calibration once on one DCU."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
import uuid

from render_treasury_arithmetic_retry import ORIGINAL_CONTROLLER_SHA256, rerender
from submit_treasury_arithmetic_once import DOMAIN, G, HOST, SSH_HELPER_SHA256, TRAIN_NATIVE, pin
from treasury_format_native import PREPARATION_RECEIPT_SHA256, PROMPTS_SHA256, load_pairs


LABEL = "training-qwen-treasury-prompt-format-calibration-v1"
ARITHMETIC = "training-qwen-treasury-dev-arith128-v1"
OLD = "training-qwen-treasury-dev-pair-hip-v2"


def make_controller(raw):
    legacy = raw.decode("utf8")
    start = legacy.index("    require(pin(PREP / \"outcome.json\")")
    end = legacy.index('    require(pin(RUN / "src/MODEL_ASSETS.json")', start)
    asset_prep = legacy[start:end]
    start = legacy.index('    for name, identity in claim["adapter_files"]')
    end = legacy.index("\n\ndef start_step", start)
    adapter_prep = legacy[start:end]
    start = legacy.index('    sys.argv = [str(RUN / "src/dev_pair_generation.py")')
    end = legacy.index("    from runpy import run_path", start)
    worker_args = legacy[start:end]
    return rerender(raw, ORIGINAL_CONTROLLER_SHA256, {
        f'RUN = RESULTS / "{OLD}"': f'RUN = RESULTS / "{LABEL}"',
        'PREP = RESULTS / "training-qwen-treasury-dev-native-CPU-v1"': 'PREP = RUN / "calibration"',
        'ADAPTER = RESULTS / "training-qwen-treasury-sft-v3/continuous/train/checkpoint-5"\n': '',
        asset_prep: ('    require(pin(PREP / "receipt.json")["SHA256"] == claim["calibration_receipt_SHA256"],\n'
                     '            "calibration preparation differs")\n'),
        adapter_prep: '',
        '(PREP, PREP, "ro"), (ADAPTER, ADAPTER, "ro"),': '',
        worker_args: (
            '    sys.argv = [str(RUN / "src/treasury_format_native.py"), "--inputs", str(PREP / "generation"),\n'
            '                "--receipt", str(PREP / "receipt.json"), "--model", str(MODEL),\n'
            '                "--output", str(RUN / "outputs"), "--deadline", str(deadline)]\n'
        ),
        'receipt["base"]["rows"] == receipt["domain"]["rows"] == 108': (
            'receipt["split"] == "train" and receipt["actual_updates"] == 0\n'
            '                    and receipt["original"]["rows"] == receipt["explicit"]["rows"] == 19'
        ),
    })


def make_host():
    source = rerender(HOST.encode(), sha256(HOST.encode()).hexdigest(), {
        f"'1087656.298': '{OLD}-gpu'": f"'1087656.315': '{ARITHMETIC}-gpu'",
        f"'1087656.299': '{OLD}-post'": f"'1087656.316': '{ARITHMETIC}-post'",
        "    return os.spawnv": (
            "    for name, (signature, blob) in DATA.items():\n"
            "        raw = base64.b64decode(blob, validate=True)\n"
            "        path = root / 'calibration' / name\n"
            "        if not path.resolve().is_relative_to(root / 'calibration') or sha256(raw).hexdigest()!=signature:\n"
            "            raise ValueError('calibration payload pin differs')\n"
            "        path.parent.mkdir(parents=True, exist_ok=True)\n"
            "        with path.open('xb') as f:\n"
            "            f.write(raw)\n"
            "    return os.spawnv"
        ),
    })
    return source


def collect_data(directory):
    receipt = directory / "receipt.json"
    proof = json.loads(pin(receipt.read_bytes(), PREPARATION_RECEIPT_SHA256))
    load_pairs(directory / "generation", receipt)
    inputs = {"receipt.json": PREPARATION_RECEIPT_SHA256, "generation/prompts.jsonl": PROMPTS_SHA256}
    inputs.update({"generation/media/" + k: v for k, v in proof["media_SHA256"].items()})
    return {name: (digest, base64.b64encode(pin((directory / name).read_bytes(), digest)).decode("ascii"))
            for name, digest in inputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    parser.add_argument("ssh_helper", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.publication):
        raise ValueError("expected an exact published commit")
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + G.as_posix(), "-C", str(G)]

    def from_git(path):
        return subprocess.check_output([*git, "show", args.publication + ":" + path], timeout=10)

    remote = subprocess.check_output([*git, "ls-remote", "--exit-code", "origin",
                                     "refs/heads/bw1000-experiments"], timeout=15).split()[0].decode()
    if remote != args.publication:
        raise ValueError("expected the exact published claim")
    claim = json.loads(from_git("coordination/tasks/" + LABEL + ".json"))
    if (claim["label"], claim["original_label"], claim["parent"], claim["node"], claim["deadline_seconds"],
        claim["max_gpu_invocations"], claim["calibration_cases"], claim["calibration_receipt_SHA256"]) != (
            LABEL, ARITHMETIC, "1087656", "a01r4n14", 2700, 1, 19, PREPARATION_RECEIPT_SHA256):
        raise ValueError("calibration scope differs")
    files = {name: from_git(DOMAIN + name) for name in (
        "treasury_format_native.py", "dev_pair_generation.py", "check_qwen_holdout_native.py",
        "check_qwen_native_inputs.py", "prepare_qwen_inputs.py", "treasury_holdout.py")}
    files.update({name: from_git(TRAIN_NATIVE + name) for name in ("snapshot.py", "slurm_accounting.py")})
    files["dev_pair_controller.py"] = make_controller(from_git(DOMAIN + "dev_pair_controller.py"))
    expected = {k: v for k, v in claim["source_SHA256"].items() if k != "MODEL_ASSETS.json"}
    if expected != {k: sha256(v).hexdigest() for k, v in files.items()}:
        raise ValueError("published source pins differ")
    host = pin(make_host(), claim["host_staging_SHA256"])
    sources = {name: base64.b64encode(raw).decode("ascii") for name, raw in files.items()}
    data = collect_data(args.data)
    if claim["generation_payload_members"] != len(data):
        raise ValueError("calibration payload member count differs")
    pin(args.ssh_helper.read_bytes(), SSH_HELPER_SHA256)
    ssh = runpy.run_path(str(args.ssh_helper))["ssh"]
    request = uuid.uuid4().hex
    header = f"CLAIM={claim!r}\nREQUEST={request!r}\nPUBLICATION={args.publication!r}\nSOURCES={sources!r}\nDATA={data!r}\n"
    body = header.encode("utf8") + host
    directory = G / "local-only" / (LABEL + "-dispatch")
    directory.mkdir()
    with (directory / "request.json").open("x", encoding="ascii") as f:
        json.dump(dict(request=request, publication=args.publication, body_SHA256=sha256(body).hexdigest()), f)
    print(json.dumps(dict(stage="request", request=request)), flush=True)
    with (directory / "stdout").open("xb") as stdout, (directory / "stderr").open("xb") as stderr:
        try:
            rc = subprocess.run([*ssh, "/usr/local/python3.12/bin/python3 -I -S -B -"],
                                input=body, stdout=stdout, stderr=stderr, timeout=2730).returncode
        except subprocess.TimeoutExpired:
            rc = None
    print(json.dumps(dict(request=request, ssh_rc=rc,
                          stdout_SHA256=sha256((directory / "stdout").read_bytes()).hexdigest(),
                          stderr_SHA256=sha256((directory / "stderr").read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
