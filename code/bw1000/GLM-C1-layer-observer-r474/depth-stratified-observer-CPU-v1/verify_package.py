"""Admit exactly the published depth-observer CPU build."""
import hashlib
import json
from pathlib import Path
import re


def verify(directory):
    root = Path(directory)
    manifest = json.loads((root / "SOURCE_FILES.json").read_bytes())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual == set(manifest) | {"SOURCE_FILES.json"}
    for name, pin in manifest.items():
        path = root / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["SHA256"]

    plan = json.loads((root / "PLAN.json").read_bytes())
    claim = json.loads((root / "CLAIM.json").read_bytes())
    baseline = json.loads((root / "BASELINE_BINDING.json").read_bytes())
    units = plan["units"]
    names = [f"u{i:02d}" for i in range(32)]
    affected = ["u00", "u09", "u10", "u11", "u12", "u15", "u17", "u18"]
    assert [unit["name"] for unit in units] == names and set(baseline["units"]) == set(names)
    assert [unit["index"] for unit in units] == list(range(32))
    assert [name for name in names if baseline["units"][name]["affected"]] == affected
    assert plan["affected_units"] == claim["affected_units"] == affected
    assert len({unit["source"] for unit in units}) == 32
    assert all(unit["kind"] == "HOST" for unit in units if unit["name"] in affected)
    assert sum(unit["kind"] == "GPU" for unit in units) == 7
    assert plan["expected_totals"] == claim["expected_totals"] == {
        "freshM": 32, "HOST_compile": 8, "GPU_compile": 0, "reused_objects": 24, "objects": 32, "links": 1,
    }
    assert claim["CPU"] == 4 and claim["memory_GiB"] == 16 and claim["max_invocations"] == 1
    assert [claim[k] for k in ("whole_epoch_seconds", "inside_from_origin_seconds",
                                "work_from_origin_seconds")] == [1800, 1740, 1680]
    assert all(claim[key] is True for key in ("ROOT_GO", "CPU_GO", "Native_compile_GO"))
    assert all(claim[key] is False for key in ("GPU_GO", "Model_GO", "Native_dlopen"))
    assert claim["arms"] == ["B"]
    assert baseline["step"] == "1103606.179" and baseline["library"] == plan["baseline_B"]
    assert baseline["build_SHA256"] == plan["baseline_B_build_SHA256"]
    assert len(plan["original169"]["object_order"]) == 32 and len(plan["original169"]["providers"]) == 71
    assert len(set(plan["prepare_aliases"])) == len(plan["prepare_aliases"]) == 11
    assert all(plan["project_pins"][unit["source"]] == unit["source_SHA256"] for unit in units)
    assert "-DGLM_R474_DEPTH_STRATIFIED_DETAILS=1" in plan["new_common_flags"]["HOST"]
    for digest, name in plan["source_blobs"].items():
        assert re.fullmatch("[a-f0-9]{64}", digest) and manifest[name]["SHA256"] == digest
    for path, old_sha in plan["old_B_project_pins"].items():
        assert plan["project_pins"][path] != old_sha and plan["project_pins"][path] in plan["source_blobs"]
        if path.endswith("/glm_sparse_observer_r131.hpp"):
            sibling = path.rsplit("/", 1)[0] + "/depth_detail_plan.hpp"
            assert plan["project_pins"][sibling] in plan["source_blobs"]
    return {"source_schema_passed": True, "planned_M": 32, "new_HOST_objects": 8, "reused_B_objects": 24}
