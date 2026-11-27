"""Production recipe and renderer checks, with synthetic negative evidence only."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from render_owner import render_owner
from render_worker import command_plan, render_worker
from scope_contract import INPUT_SHA, ORIGINAL_ARCHIVE_SHA, audit_ranks, sha, view_bytes

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORE = HERE.parent / "training-minicpm-usgs-exposure25-resume20-hip-v1"
TARGET = HERE.parent / "training-minicpm-usgs-exposure25-resume20-target-v2"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reject(action, code):
    try:
        action()
    except ValueError as error:
        assert str(error) == code
    else:
        raise AssertionError(code)


def fixture_ranks(request):
    base = dict(SHA256="1"*64, tensors=4, elements=10822656, floating_elements=10822656,
                floating_l2=1.0, finite=True)
    changed = dict(base, SHA256="2"*64, floating_l2=1.1)
    updates = []
    for i, rate, after, movement in ((1, 0.0, base, 0), (2, 2e-5, changed, 1)):
        delta = dict(l2=float(movement), max_abs=float(movement), parameter_tensors=4,
                     changed_elements=movement, changed_tensors=movement)
        updates.append(dict(global_step=i, actual_optimizer=dict(step=i, lr=[rate], before=base, after=after, delta=delta),
                            optimizer_step_min=i, optimizer_step_max=i, optimizer_slot_count=4))
    inputs = [dict(ordinal=i+1, global_step=i//4, tokens={"shifted_supervised_tokens":6}) for i in range(8)]
    outputs = [dict(ordinal=i+1, loss_scalar=1.0+i/10) for i in range(8)]
    return [deepcopy(dict(request=request, rank=i, world_size=8, completed_global_step=2, saved_checkpoints=[2],
                         inputs=inputs, outputs=outputs, steps=updates, checkpoint_adapter=changed)) for i in range(8)]


def main():
    req = "0a96e74bdef149488fba89065a185ff0"
    scope = dict(request=req, label="training-minicpm-off-native2-"+req[:12], parent="99999999", node="f0r0n0", port=31012)
    recipe = load(CORE / "minicpm_domain_recipe.py")
    validator = load(HERE.parent / "training-minicpm-off-native6-CPU-v1/off_inputs.py")
    bundle = (REPO / "local-only/minicpm-off-native6-offline-fixture-20261127/train").resolve(strict=True)
    rows = validator.read_inputs(bundle, INPUT_SHA)
    view, ledger = view_bytes(rows)
    assert len(view.splitlines()) == 800 and ledger["exposure_counts"] == [134, 134, 133, 133, 133, 133]
    assert [json.loads(line)["messages"] for line in view.splitlines()[:6]] == [row["messages"] for row in rows]
    assert len({hash for hash in ledger["original_derived_row_SHA256"]}) == 6
    run = "/var/fixture/" + scope["label"]
    translator = HERE.parent / "training-qwen-native-update-contract-v1/deltas/tools/hygon_vlm_recipe.py"
    plan = command_plan(recipe, translator, (CORE/"original-launch.json").read_bytes(), run, sha(view), scope["port"])
    assert plan["command"][0] == run+"/framework/vlm/runtime/swift_entry.py"
    assert plan["planned_optimizer_updates"] == 2 and plan["nominal_max_steps"] == 25
    assert plan["required_assets"]["plugins"][-1] == run+"/src/off_callback.py" or run+"/src/off_callback.py" in plan["required_assets"]["plugins"]
    worker_bytes = render_worker((TARGET/"worker.py").read_bytes(), scope, "b"*64)
    owner_bytes = render_owner((REPO/"tools/control_minicpm_usgs_pair_target_v3.py").read_bytes(), scope, "b"*64)
    assert b"ACCOUNTING_HELPER = RUN/'src'/'owner-deps'" in owner_bytes and b"OFF_OWNER_DEPENDENCY" in owner_bytes
    for name, raw in (("worker", worker_bytes), ("owner", owner_bytes)):
        ast.parse(raw, filename=name, feature_version=(3, 10))
        assert b"training-domain-usgs-table-reading-v1" not in raw
    owner_ns = {"__name__":"fixture_owner","__file__":str(HERE/"renders/owner.py")}
    # Execute the actual generated pure functions with only import and filesystem roots
    # pointed to the local checkout. No subprocess command is called.
    owner_tree = ast.parse(owner_bytes)
    functions = [node for node in owner_tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("exact_tres", "step_command")]
    exec(compile(ast.Module(body=functions,type_ignores=[]), "rendered-pure-owner", "exec"),owner_ns)
    owner_ns.update(dict(PHASES={"pre":(0,2,2048,60),"gpu":(8,64,224000,2100),"post":(0,2,2048,90)},
                         PARENT="99999999",NODE="f0r0n0",LABEL=scope["label"],
                         need=lambda m,n: None if m else (_ for _ in ()).throw(ValueError(n)),
                         math=__import__("math")))
    assert owner_ns["exact_tres"]("cpu=64,gres/dcu=8,mem=224000M,node=1","gpu")
    assert not owner_ns["exact_tres"]("cpu=2,gres/dcu=1,mem=2G,node=1","pre")
    assert "--gres=dcu:8" in owner_ns["step_command"]("gpu", ["no-execution"])
    evidence = fixture_ranks(req)
    assert audit_ranks(evidence,req)["native_forwards"] == 64
    reject(lambda: audit_ranks(evidence[:-1],req),"OFF_RANK_DENOMINATOR")
    evidence[7]["steps"][1]["actual_optimizer"]["lr"] = [0]
    reject(lambda: audit_ranks(evidence,req),"OFF_ACTUAL_LR")
    evidence = fixture_ranks(req)
    evidence[7]["steps"][1]["actual_optimizer"]["after"]["SHA256"] = "3"*64
    reject(lambda: audit_ranks(evidence,req),"OFF_EIGHT_RANK_UPDATE_AGREEMENT")
    archive = (REPO / "results/bw1000/2026-11-27/specialized-minicpm-OFF-native6-4479-independent-zero-GRES.json").read_bytes()
    assert sha(archive) == ORIGINAL_ARCHIVE_SHA and json.loads(archive)["actual"]["independent"]["passed"] is True
    print(json.dumps(dict(status="PASS", view_SHA256=sha(view), view_bytes=len(view), rows=6, unique_media=4,
                         physical_rows=800, source_recipe_checked=True, renderer_entries=2, actual_Torch_training=0,
                         synthetic_rank_fixture_only=8), sort_keys=True))


if __name__ == "__main__":
    main()
