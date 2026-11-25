"""Continue the exact completed T07 fallback through the existing guarded sender."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
from types import FunctionType


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
STEM = "training80-t07-3081-guarded-bf16-v1"
CLAIM = REPO / "coordination/tasks" / (STEM + ".json")
BASE = REPO / "tools/run_training80_prebridge_five_guarded_refill_once.py"
BASE_SHA = "7b88d30a4af7eff4f25fbcc663de674af328a03de74e364c517752bd599d306c"
NODE = "f13r2n07"
PREVIOUS, CLEAR = "1103606.3081", "1103606.3080"
OLD_REQUEST = "b3ce4e40bc664e32ab74ce203cb8eeff"
NEW_REQUEST = "df982d1cc29d4ab39515b752eea2c164"
PROOF = LOCAL / "training80-T07-original3081-terminal-dual-source-20261125.json"
PROOF_SHA = "82a8c53449660601600c4bd01b3ae4d59eedb142ddb1e04f643f067ff829198d"
FROZEN_PRIOR = {
    "expected": "04924af56c5e654bf7ad652031d7fb7153660f63d4794fe79c93b7fd4ab59494",
    "worker": "06503f4598333761a6638d110780c2f248ede71f35e14d0bdcb2878aa3d8a045",
    "terminal": "cf6f6c226bfb1c1e6bde663b085098acc2a282a6f4341a192b9aea722f72ffec",
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def load():
    need(digest(BASE.read_bytes()) == BASE_SHA, "frozen guarded base source changed")
    base = runpy.run_path(str(BASE))
    config = base["prepared"](json.loads(base["pin"](base["CLAIM"], "c4c18f644e6ebc2a99ea0e19d2261a9cbbe47efe743e79dd5ae8f79fea4ce974")))
    raw = base["pin"](CLAIM)
    claim = json.loads(raw)
    need(claim == {
        "scope": STEM, "node": NODE, "parent": "1103606", "request": NEW_REQUEST,
        "previous_request": OLD_REQUEST, "previous_gpu_step": PREVIOUS, "previous_zero_GRES": CLEAR,
        "previous_terminal_proof_SHA256": PROOF_SHA, "prior_worker_pins": FROZEN_PRIOR,
        "GPU": 8, "maximum_invocations": 1, "replay_allowed": False,
    }, "fixed T07 scope differs")
    proof = json.loads(base["pin"](PROOF, PROOF_SHA))
    actual = proof["observed"]
    need(proof["rc"] == 0 and actual["request"] == OLD_REQUEST and actual["owner"] and actual["alloc_320"]
         and actual["parent_state"] == "RUNNING" and actual["live"] == actual["squeue"] == {},
         "actual original T07 absence is unverified")
    gpu, = actual["accounting"][PREVIOUS]
    cpu, = actual["accounting"][CLEAR]
    need(base["terminal"](gpu, PREVIOUS, NODE, base["GPU_TRES"])
         and base["terminal"](cpu, CLEAR, NODE, base["HOST_TRES"]), "actual predecessor terminal TRES differ")
    for name, pin in FROZEN_PRIOR.items():
        need(actual[name]["SHA256"] == pin, "original worker receipt differs")
    e, w, t = (actual[name]["document"] for name in ("expected", "worker", "terminal"))
    pci = e["pci"]
    need(e["request"] == OLD_REQUEST and e["label"] == gpu[1] and len(pci) == len(set(pci)) == 8
         and e["node"] == w["node"] == t["node"] == NODE and w["step"] == t["step"] == PREVIOUS
         and pci == w["pci"] and w["synthetic"] is True and t["state"] == "WORKER_RETURNED"
         and e["seconds"] == 10560 and w["epoch"] + e["seconds"] <= t["epoch"] < actual["epoch"],
         "original T07 worker did not return normally")
    config[3][NODE] = (gpu, cpu, pci)
    return base, config, digest(raw)


def function(base, name):
    src = base[name]
    return FunctionType(src.__code__, {**src.__globals__, "STEM": STEM, "FIVE": {NODE: (PREVIOUS, CLEAR)},
                                      "paths": FunctionType(base["paths"].__code__,
                                                            {**base["paths"].__globals__, "STEM": STEM})})


def render(base, config, publication):
    program = function(base, "render")(NODE, {"targets": {NODE: {"request": NEW_REQUEST}}}, config, publication)
    boundary = "            check_previous(account(PREVIOUS))"
    need(program["launch"].count(boundary) == 1, "original lock boundary changed")
    previous_dir = "training80-dcu-standby-" + OLD_REQUEST
    gate = (
        "            for member, expected_sha in " + repr(FROZEN_PRIOR) + ".items():\n"
        f"                old_file = RESULTS / {previous_dir!r} / NODE / (member + '.json')\n"
        "                require(old_file.is_file() and old_file.resolve(strict=True) == old_file, 'original artifact path changed')\n"
        "                with old_file.open('rb') as handle:\n"
        "                    prior_raw = handle.read(8193)\n"
        "                require(0 < len(prior_raw) <= 8192 and sha256(prior_raw).hexdigest() == expected_sha, 'original worker terminal changed')\n"
    )
    prefix = "import os\nos.environ.update(TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')\n"
    program["launch"] = prefix + program["launch"].replace(boundary, gate + boundary)
    program["observe"] = prefix + program["observe"]
    for name in ("launch", "observe"):
        compile(program[name], "<exact T07 continuation>", "exec")
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--publication")
    modes.add_argument("--observe-original", action="store_true")
    args = parser.parse_args()
    base, config, claim_sha = load()
    prog = render(base, config, args.publication)
    if args.check:
        result = dict(request=NEW_REQUEST, predecessor=PREVIOUS, worker_seconds=prog["worker_seconds"],
                      claim_SHA256=claim_sha, launch_SHA256=digest(prog["launch"].encode()), new_GPU_steps=0)
    elif args.observe_original:
        result = function(base, "observe")(NODE, prog, config)
    else:
        base["verify_publication"](args.publication, config)
        git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
        for path in (Path(__file__).resolve(), CLAIM):
            ref = args.publication + ":" + path.relative_to(REPO).as_posix()
            need(subprocess.check_output([*git, "cat-file", "blob", ref], cwd=REPO, timeout=10) == path.read_bytes(),
                 "published T07 source or claim differs")
        ledger_claimed = base["live_authority"](config)
        result = function(base, "dispatch")(NODE, prog, args.publication, claim_sha, config, ledger_claimed)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
