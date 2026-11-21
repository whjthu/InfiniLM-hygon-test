"""Close the original Graph64 v2 shards with the frozen per-shard verifier."""

from hashlib import sha256
import json
from pathlib import Path

from prepare_training80_vggt_direct_finish import bounded
from report_training80_backlog import REPO, need


REQUEST = "5a3b2a0fa2f504b4200e7de2738d42a0"
PACKAGE = Path("code/bw1000/fleet-bf16-projection-graph64-backfill-v2")
DIRECTORY = Path("local-only") / ("O01-resident-BF16-graph64-fleet-v2-" + REQUEST + "-original")
PLAN_SHA = "948bd7e485f6ca45bf653c249207987abbe2caad7ca141eecc427ea3f3e60e5a"
CLAIM_SHA = "dfd43c1ed1224aed87f5e5a892ecb693cdf49e2c7a8020aae59bbd20f73c8895"
SCOPE_SHA = "79154688484daad4c712360b6fe6be3edea1f15cf983eceb0ca1c1e96a8a0030"
CONTROLLER_SHA = "3a3575da7a11fb5b8e4456b13c6670676b8475cf13a33c6c6b9a71c8333094d9"
SOURCE_MANIFEST_SHA = "caf369cc5f7dfaa39ed68573b9e3ab02bbf33c11cbdd04993beacefcedc17439"
PUBLICATION = "9845fa0d3b9c66871c1ed2c82930c4d000227ae0"
VERIFIER_SHA = "d72afea2a45196f186adc00ff8aaa47895c983b1bcb0d18fc0c7fae56cb2cc88"


def sha(raw):
    return sha256(raw).hexdigest()


def frozen_source(repo):
    claim = bounded(repo / PACKAGE / "ROOT_CLAIM.json", 8192)
    manifest = bounded(repo / PACKAGE / "SOURCE_FILES.json", 16384)
    need(sha(claim) == CLAIM_SHA and json.loads(claim)["request"] == REQUEST and sha(manifest) == SOURCE_MANIFEST_SHA,
         "original Graph64 v2 claim or source manifest differs")
    refs = json.loads(manifest)["files"]
    pins = {}
    for name, spec in refs.items():
        relative = Path(spec["path"])
        need(not relative.is_absolute() and ".." not in relative.parts, "invalid Graph64 v2 manifest path")
        path = repo / relative
        need(path.resolve(strict=True) == path, "noncanonical Graph64 v2 manifest member")
        raw = bounded(path, 32768)
        need(len(raw) == spec["bytes"] and sha(raw) == spec["SHA256"], "Graph64 v2 source member differs: " + name)
        pins[name] = spec["SHA256"]
    pins["SOURCE_FILES.json"] = SOURCE_MANIFEST_SHA
    return sha(json.dumps(dict(sorted(pins.items())), separators=(",", ":")).encode())


def bound_verifier():
    path = Path(__file__).resolve().with_name("close_graph64_original_backfill.py")
    source = bounded(path, 65536)
    need(sha(source) == VERIFIER_SHA, "frozen Graph64 per-shard verifier bytes differ")
    # Use an isolated namespace so this invocation never changes v1's original campaign.
    checker = {"__name__": "_graph64_v2_original_verifier", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), checker)
    checker.update(REQUEST=REQUEST, PACKAGE=PACKAGE, DIRECTORY=DIRECTORY, PLAN_SHA=PLAN_SHA,
                   CLAIM_SHA=CLAIM_SHA, SCOPE_SHA=SCOPE_SHA, CONTROLLER_SHA=CONTROLLER_SHA, PUBLICATION=PUBLICATION)
    read_original = checker["original_inputs"]

    def original_inputs(repo):
        originals = read_original(repo)
        need(originals[0]["source_SHA256"] == frozen_source(repo), "v2 plan does not bind the original source envelope")
        return originals

    checker["original_inputs"] = original_inputs
    return checker


_checker = bound_verifier()
terminal_receipt = _checker["terminal_receipt"]
original_inputs = _checker["original_inputs"]
close = _checker["close"]


if __name__ == "__main__":
    _checker["main"]()
