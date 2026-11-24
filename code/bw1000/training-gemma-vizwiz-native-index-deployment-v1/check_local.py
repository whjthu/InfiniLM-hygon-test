"""Verify the adopted index deployment command and original evidence bindings."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import sys

import index_entry as entry
import render
import verify_terminal


ROOT = Path(__file__).resolve().parents[3]
EPOCH = 1795500000
NODE = "f99r9n99"
FIXTURE = str(render.PROJECT / "incoming/fixture-vizwiz-index-no-dispatch")


def reject(call, gate):
    try:
        call()
    except ValueError as exc:
        assert str(exc) == gate, exc
    else:
        raise AssertionError("not rejected")


def verify_original_pins():
    base = ROOT / "results/bw1000/2026-11-25"
    names = ("specialized06-vizwiz-five-native-input-original2882-terminal.json",
             "specialized06-vizwiz-reviewed-seven-original2883-native-input.json")
    records = []
    for filename, group, source in zip(names, ("verified", "terminal"), entry.ORIGINAL_WORKERS):
        data = (base / filename).read_bytes()
        documented = json.loads(data)[group]
        _, size, digest, step, result_size, result_sha, _, count = source
        assert documented.get("row", documented.get("accounting_row"))[0] == step
        assert documented.get("outcome", documented.get("original_outcome")) == {
            "bytes": result_size, "SHA256": result_sha}
        assert documented["native_rows"] == count and documented["worker"] == {"bytes": size, "SHA256": digest}
        records.append({"SHA256": sha256(data).hexdigest(), "native_CPU_step": step, "pinned_original_train_rows": count})
    return records


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected one absent private receipt")
    files = [Path(entry.__file__), Path(render.__file__), Path(verify_terminal.__file__)]
    helper = ROOT / "code/bw1000/training-gemma-vizwiz-two-updates-source-v1/gemma_vizwiz_two_updates_inputs.py"
    assert entry.pin(helper.read_bytes()) == {"bytes": entry.INDEX_HELPER[0], "SHA256": entry.INDEX_HELPER[1]}
    for file in files:
        ast.parse(file.read_bytes())
        compile(file.read_bytes(), str(file), "exec")
    argv = render.compute_command("1103606", NODE, EPOCH, FIXTURE + "/code", FIXTURE + "/view", FIXTURE + "/out")
    assert argv.count("--gres=none") == 1 and "/candidate-source/index_entry.py" in argv[-1]
    assert "CUDA_VISIBLE_DEVICES=-1" in argv[-1] and "HF_HUB_OFFLINE=1" in argv[-1]
    assert sum(item.endswith(":rw") for item in argv) == 1 and "--rocm" not in argv
    assert all(sum(1 for text in argv if text == f"{render.RESULTS}/{stage}:{render.RESULTS}/{stage}:ro") == 1
               for stage, _, _ in entry.STREAMS)
    assert sum(item.endswith("outcome.json:ro") for item in argv) == 2
    env = {"SLURM_JOB_ID": "1103606", "SLURM_STEP_ID": "7", "SLURM_CPUS_PER_TASK": "4",
           "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert entry.bind_cpu("1103606", NODE, EPOCH, EPOCH + 1, env, NODE, 4) == "1103606.7"
    for extra in ({"SLURM_STEP_GPUS": "0"}, {"SLURM_TRES_PER_TASK": "cpu=4,gres/dcu=1"},
                  {"CUDA_VISIBLE_DEVICES": "0"}):
        reject(lambda: entry.bind_cpu("1103606", NODE, EPOCH, EPOCH+1, dict(env, **extra), NODE, 4),
               "NO_GPU_OFFLINE_STEP")
    line = f"1103606.7|{render.LABEL}|COMPLETED|0:0|2|{NODE}|cpu=4,mem=8G,node=1"
    assert render.terminal(line, "1103606", NODE, "1103606.7", require_success=True)[6] == "cpu=4,mem=8G,node=1"
    reject(lambda: render.terminal(line + ",gres/dcu=1", "1103606", NODE, "1103606.7"), "ACTUAL_ZERO_GRES_ACCOUNTING")
    reject(lambda: render.terminal(line + "\n" + line, "1103606", NODE, "1103606.7"), "EXACT_INDEPENDENT_ACCOUNTING")
    # Identity/counter passthrough uses sentinel objects, without simulating Torch or native execution.
    class Delegate:
        def encode(self, data):
            return data

        def data_collator(self, _rows):
            raise ValueError("fixture delegate failure")

    progress = {"native_encodes_completed": 0, "native_collations_completed": 0}
    counted = entry.CountedTemplate(Delegate(), progress, lambda: None)
    assert counted.encode("fixture") == "fixture"
    reject(lambda: counted.data_collator(["fixture"]), "fixture delegate failure")
    assert progress == {"native_encodes_completed": 1, "native_collations_completed": 0}
    failure = {"state": "FAILED", "CPU_step": None, "planned_train_rows": 12, "qualified_index_rows": 0,
               "calibration_roles": 0, "validation_roles": 0, "test_roles": 0, "model_loads": 0,
               "optimizer_updates": 0, "GPU": 0, **progress}
    rejected_row = line.replace("|COMPLETED|0:0|", "|FAILED|1:0|")
    retained = verify_terminal.verify(rejected_row, "1103606", NODE, "1103606.7", failure, None)
    assert retained["planned_train_rows"] == 12 and retained["qualified_index_rows"] == 0
    assert retained["native_encodes_completed"] == 1 and retained["native_collations_completed"] == 0
    reject(lambda: verify_terminal.verify(rejected_row, "1103606", NODE, "1103606.7", failure, b"fixture"),
           "FAILED_INDEX_PRESENT")
    assert len(verify_terminal.published_lines()) == 12
    receipt = {"state": "LOCAL_SOURCE_AND_ARGV_VERIFIED", "early_failure_test": retained, "argv_capture": argv,
               "source_pins": {file.name: entry.pin(file.read_bytes()) for file in files},
               "original_collator_evidence": verify_original_pins(),
               "planned_train_rows": 12, "published_800_view_SHA256": entry.VIEW, "actual_qualified_index_rows": 0,
               "native_CPU_calls": 0, "new_Slurm_calls": 0, "model_loads": 0, "optimizer_updates": 0, "GPU": 0,
               "calibration_roles": 0, "validation_roles": 0, "test_roles": 0}
    print(json.dumps(entry.save(Path(sys.argv[1]), receipt)))


if __name__ == "__main__":
    main()
