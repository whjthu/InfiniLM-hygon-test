"""Offline checks for exact wire, process isolation and timeline evidence boundaries."""

import ast
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import euler as e


def rejects(fn, reason):
    try:
        fn()
    except ValueError as error:
        assert str(error) == reason, str(error)
    else:
        raise AssertionError(reason)


def projection(raw, shape, dtype):
    return dict(dtype=dtype, shape=shape, bytes=len(raw), raw_hex=raw.hex(), SHA256=e.digest(raw))


def independent_seeds(scratch):
    seen = []
    for seed in e.SEEDS:
        argv, env = e.command(Path(e.__file__).resolve(), scratch, scratch / seed, seed)
        assert "--seed" in argv and argv[-1] == env["PYTHONHASHSEED"] == seed
        raw = subprocess.check_output([argv[0], "-B", "-c",
            "import json,os; print(json.dumps([os.getpid(),os.environ['PYTHONHASHSEED'],hash('sdxl-euler-order-controlled')]))"],
            env=env)
        seen.append(json.loads(raw))
    assert len({row[0] for row in seen}) == len({row[2] for row in seen}) == 2
    assert [row[1] for row in seen] == list(e.SEEDS)
    return seen


def ledger_fixture(root, raw_sha, timeline):
    files = root / "original"
    files.mkdir()
    ledger = b"".join(e.wire(dict(arm=arm, pair=i, outcome="completed", timesteps_SHA256=timeline))
                      for arm in ("base", "cp25") for i in range(6))
    summary = dict(request=e.OLD_REQUEST, plan_SHA256=e.PLAN, planned_pairs=6, complete_same_noise_pairs=6,
                   planned_slots=12, sealed=True, passed=True, generations_sha256=e.digest(ledger),
                   generations_bytes=len(ledger), facts=dict(step="1194900.2762", effective_scheduler_config_SHA256=raw_sha))
    summary["generations_SHA256"] = summary.pop("generations_sha256")
    (files / "result.json").write_bytes(e.wire(summary))
    (files / "generations.jsonl").write_bytes(ledger)
    with patch.object(e, "OLD_RESULT", e.digest(e.wire(summary))):
        assert e.original(files, readonly=False) == ([timeline] * 6, e.digest(ledger))
        (files / "generations.jsonl").write_bytes(ledger + b" ")
        rejects(lambda: e.original(files, readonly=False), "EXACT_STABLE_SOURCE_BYTES")
    return e.digest(ledger)


def comparisons(root):
    observed = independent_seeds(root)
    first = {e.FIELD: ["a", "b", "c"], "timestep_spacing": "linspace", "steps_offset": 1}
    second = dict(first, **{e.FIELD: ["b", "c", "a"]})
    baseline = e.digest(e.wire(dict(first, **{e.FIELD: ["c", "b", "a"]})))
    timesteps = projection(b"x" * 120, [30], "torch.float32")
    sigma = projection(b"s" * 124, [31], "torch.float32")
    default = projection(b"g" * 4004, [1001], "torch.float32")
    with patch.object(e, "OLD_WIRE", baseline):
        ledger_sha = ledger_fixture(root, baseline, timesteps["SHA256"])
        assert e.reconstruct(first, baseline)["found"]
        capped = e.reconstruct(first, baseline, limit=2)
        assert not capped["found"] and not capped["exhaustion"] and capped["attempts"] == 2
        wrong = e.reconstruct(dict(first, steps_offset=2), baseline)
        assert not wrong["found"] and wrong["exhaustion"] and wrong["attempts"] == 6

        def receipt(process, config, time=timesteps, sigma_value=sigma):
            return dict(pid=process[0], seed=process[1], python_hash=process[2], gpu_initialized=False,
                        scheduler_source_SHA256=e.EULER_PROVIDER, source_config_SHA256=e.SCHEDULER,
                        diffusers_version="0.39.0", torch_version="fixture",
                        config=config, raw_wire_SHA256=e.digest(e.wire(config)),
                        timesteps=time, sigmas=sigma_value, default_sigmas=default)

        a, b = receipt(observed[0], first), receipt(observed[1], second)
        report = e.compare([a, b], [timesteps["SHA256"]] * 6, ledger_sha)
        assert report["same_default_member_order_drift_observed"] and all(
            all(matches) for matches in report["original_six_timestep_matches"])
        assert report["original_sigma_recorded"] is False and report["original_2774_cause_proven"] is False
        assert all(report["tensor_values_equal_between_children"].values())
        changed = receipt(observed[1], dict(second, steps_offset=True), time=projection(b"d" * 120, [30], "torch.float32"))
        drift = e.compare([a, changed], [timesteps["SHA256"]] * 6, ledger_sha)
        assert not drift["normalized_equal"] and drift["default_order_changed"]
        assert not drift["tensor_values_equal_between_children"]["timesteps"] and not any(
            drift["original_six_timestep_matches"][1])
        changed["timesteps"]["raw_hex"] = "00" * 120
        rejects(lambda: e.compare([a, changed], [timesteps["SHA256"]] * 6, ledger_sha), "EXACT_CHILD_TENSOR_BYTES")


def main():
    repo = Path(__file__).resolve().parents[3]
    receipt = repo / "local-only/sdxl-four-cp25-2774-independent-terminal-and-euler-provider-99e168f4dfa44ca094c6fb74349130ff.json"
    actual = json.loads(receipt.read_bytes())["response"]
    assert actual["configuration_file_SHA256"] == e.CONFIGURATION
    assert actual["original_effective_scheduler_SHA256"] == e.OLD_WIRE
    source = ast.parse(Path(e.__file__).read_bytes())
    child, = (node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "child")
    calls = [node for node in ast.walk(child) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert sum(node.func.attr == "from_pretrained" for node in calls) == 1
    assert sum(node.func.attr == "set_timesteps" for node in calls) == 1
    with TemporaryDirectory() as directory:
        comparisons(Path(directory).resolve())
    print("actual .2774 provider/wire pins, real Python seed isolation, six-slot ledger fixture, bounded permutation and tensor negatives PASS")


if __name__ == "__main__":
    main()
