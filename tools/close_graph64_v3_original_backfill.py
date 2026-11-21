"""Close the original six-node Graph64 v3 comparison, one proven shard at a time."""

from hashlib import sha256
import json
from pathlib import Path

from prepare_training80_vggt_direct_finish import bounded
from report_training80_backlog import REPO, hex_value, need


REQUEST = "ce9cd75da7d17f811e78e0f88a9d963d"
PACKAGE = Path("code/bw1000/fleet-bf16-projection-graph64-setup-v3")
DIRECTORY = Path("local-only") / ("O01-resident-BF16-graph64-setup-v3-" + REQUEST + "-original")
PLAN_SHA = "f29211a5bdf498137ead35755a9a725d7ecb44b9d74eda170254401734ce913b"
CLAIM_SHA = "be6bd48e1aa7c5eefe1ff14acb70641782d0c41ebc35ea45dbce40338db7b9b5"
SCOPE_SHA = "3a84b9b82cf02764fc811bbad8454e7afc12fd6c09d9e5ed0e4f5387699e4983"
CONTROLLER_SHA = "52213592da389642cc174195728533672cf67413669b129866a01d21f4578bb5"
SOURCE_MANIFEST_SHA = "21175aa7fa5fea31149e2378015f2040accb6fe405f3b19414f3dd9a45967272"
PUBLICATION = "3176c1f7523c586232ad6701fe71cb4595ccf9b8"
CASE = "O01.resident-bf16-graph64-setup-comparison"
DENOMINATOR = dict(planned_nodes=6, planned_cards=48, planned_arms=192, planned_busy_card_samples=768)
VERIFIER_SHA = "d72afea2a45196f186adc00ff8aaa47895c983b1bcb0d18fc0c7fae56cb2cc88"


def sha(raw):
    return sha256(raw).hexdigest()


def frozen_source(repo):
    claim = bounded(repo / PACKAGE / "ROOT_CLAIM.json", 8192)
    manifest = bounded(repo / PACKAGE / "SOURCE_FILES.json", 16384)
    need(sha(claim) == CLAIM_SHA and json.loads(claim)["request"] == REQUEST and sha(manifest) == SOURCE_MANIFEST_SHA,
         "original Graph64 v3 claim or source manifest differs")
    pins = {}
    for name, spec in json.loads(manifest)["files"].items():
        relative = Path(spec["path"])
        need(not relative.is_absolute() and ".." not in relative.parts, "invalid Graph64 v3 manifest path")
        path = repo / relative
        need(path.resolve(strict=True) == path, "noncanonical Graph64 v3 manifest member")
        raw = bounded(path, 32768)
        need(len(raw) == spec["bytes"] and sha(raw) == spec["SHA256"], "Graph64 v3 source member differs: " + name)
        pins[name] = spec["SHA256"]
    pins["SOURCE_FILES.json"] = SOURCE_MANIFEST_SHA
    return sha(json.dumps(dict(sorted(pins.items())), separators=(",", ":")).encode())


def bound_verifier():
    path = Path(__file__).resolve().with_name("close_graph64_original_backfill.py")
    source = bounded(path, 65536)
    need(sha(source) == VERIFIER_SHA, "frozen Graph64 per-shard verifier bytes differ")
    # Only the frozen plan-size assertion and its messages need to change in the shared proof.
    for before, after in (
        (b'len(shards) == 31 and len(plan["selected"]) == 75',
         b'len(shards) == 3 and len(plan["selected"]) == DENOMINATOR["planned_nodes"]'),
        (b'original Graph64 plan changed its 31-shard or 75-node scope',
         b'original Graph64 v3 plan changed its three-shard or six-node scope'),
        (b'original authenticated 31-shard result', b'original authenticated v3 three-shard result'),
    ):
        need(source.count(before) == 1, "frozen Graph64 proof binding differs")
        source = source.replace(before, after)
    checker = {"__name__": "_graph64_v3_original_verifier", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), checker)
    checker.update(REQUEST=REQUEST, PACKAGE=PACKAGE, DIRECTORY=DIRECTORY, PLAN_SHA=PLAN_SHA, CLAIM_SHA=CLAIM_SHA,
                   SCOPE_SHA=SCOPE_SHA, CONTROLLER_SHA=CONTROLLER_SHA, PUBLICATION=PUBLICATION, CASE=CASE,
                   DENOMINATOR=DENOMINATOR)
    read_original, verify_counts = checker["original_inputs"], checker["counters"]

    def original_inputs(repo):
        originals = read_original(repo)
        need(originals[0]["source_SHA256"] == frozen_source(repo), "v3 plan does not bind the original source envelope")
        return originals

    def counters(verdict, unit, item, row):
        passed, counts = verify_counts(verdict, unit, item, row)
        if passed:
            for node in unit["nodes"]:
                setup = verdict["nodes"][node]["setup_phase"]
                need((setup["request"], setup["node"], setup["step"]) == (REQUEST, node, row[0])
                     and setup["all_released"] is True and hex_value(setup["raw_SHA256"], 64),
                     "v3 qualified work lacks the original node-local setup gate proof")
        return passed, counts

    checker.update(original_inputs=original_inputs, counters=counters)
    return checker


_checker = bound_verifier()
terminal_receipt = _checker["terminal_receipt"]
original_inputs = _checker["original_inputs"]
close = _checker["close"]


if __name__ == "__main__":
    _checker["main"]()
