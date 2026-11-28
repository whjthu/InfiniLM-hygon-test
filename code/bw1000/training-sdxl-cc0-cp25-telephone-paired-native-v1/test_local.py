"""Test the actual sealed plan, historical source bindings and failure denominator."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from plan import ARMS, CLOCK_SHA, FAMILIES, FAMILY_SHA, digest, frozen, original_run, read, seal
from score_cpu import final_pairs


def rejects(fn, code):
    try:
        fn()
    except ValueError as error:
        assert str(error) == code, str(error)
    else:
        raise AssertionError(code)


def main():
    root = Path(__file__).resolve().parents[3]
    published = root / "results/bw1000/2026-11-28"
    original_run(published / "specialized-sdxl-2675-native-full25-cp20-exact-recovery-and-checkpoints.json")
    sources = json.loads(read(published / "specialized-two-cc0-product-families-official-bundles.json", 16 << 10, FAMILY_SHA))
    clock = json.loads(read(published / "specialized-public-cc0-painted-stool-wall-clock-source-and-atlas.json", 16 << 10, CLOCK_SHA))
    assert sources["license"] == "CC0-1.0" and sources["native_training_rows"] == 0
    for family in ("cassette_player", "vintage_electric_kettle"):
        assert sources["bundles"][family]["sha256"] == FAMILIES[family][1] and sources["bundles"][family]["independent_readback"]
    entry, = (row for row in clock["assets"] if row["asset"] == "vintage_telephone_wall_clock")
    assert entry["admission_SHA256"] == FAMILIES[entry["asset"]][1] and entry["native_training_rows"] == 0
    path = Path(__file__).resolve().with_name("PLAN.json")
    plan, plan_sha = frozen(path)
    assert len({row["prompt_SHA256"] for row in plan["samples"]}) == 3
    assert len({row["seed"] for row in plan["samples"]}) == 6
    def completed(arm, item):
        row = {key: item[key] for key in ("pair", "family", "replicate", "seed", "prompt_SHA256")}
        return dict(row, arm=arm, outcome="completed", raw_latent_SHA256="a" * 64,
                    prepared_latent_SHA256="b" * 64, timesteps_SHA256="c" * 64)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary).resolve()
        tampered = directory / "PLAN.json"
        tampered.write_bytes(path.read_bytes() + b" ")
        rejects(lambda: frozen(tampered), "EXACT_PRE_OUTCOME_PLAN")
        pairs = {(arm, 0): completed(arm, plan["samples"][0]) for arm in ARMS}
        mismatch = dict(pairs["cp25", 0], prepared_latent_SHA256="d" * 64)
        failure = completed("cp25", plan["samples"][1])
        failure["outcome"] = "error"
        pairs["cp25", 1] = failure
        witness = {"step": "1194900.9001", "fixture": True}
        first = directory / "partial"
        first.mkdir()
        report = seal(first, plan, plan_sha, "fixture", pairs, witness, {"phase": "fixture"})
        assert report["planned_slots"] == 12 and report["planned_pairs"] == 6
        assert report["complete_same_noise_pairs"] == 1 and report["passed"] is False
        assert report["counts"]["cp25"] == {"completed": 1, "error": 1, "missing": 4}
        proof = dict(request="fixture", numbered_gpu_step=witness["step"], independent_post_clean=True,
                     worker_result_SHA256=digest((first / "result.json").read_bytes()))
        assert len(final_pairs(first, plan, plan_sha, proof)[1]) == 1
        rejects(lambda: final_pairs(first, plan, plan_sha, dict(proof, independent_post_clean=False)),
                "EXACT_INDEPENDENT_TERMINAL_AND_SEAL")
        second = directory / "different-noise"
        second.mkdir()
        assert seal(second, plan, plan_sha, "fixture", dict(pairs, **{}) | {("cp25", 0): mismatch},
                    witness, None)["complete_same_noise_pairs"] == 0
    print(json.dumps(dict(original_r33_terminal=True, exact_families=3, fixed_noise_seeds=6, planned_slots=12,
                          failure_ledger=True, independent_cpu_scoring_intake=True, actual_generations=0,
                          actual_scores=0, gpu_calls=0), sort_keys=True))


if __name__ == "__main__":
    main()
