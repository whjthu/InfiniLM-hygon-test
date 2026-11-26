"""Render the exact heldout worker delta or reconcile the original private heldout output."""

import argparse
import ast
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import heldout_contract as contract


BASE = REPO / "code/bw1000/training-zimage-style18-native-dev12-pair-v1"
PINS = {
    "worker.py": dict(bytes=12745, SHA256="89b3c51fb91fea9aa27ad026ce08ea965a7c9b0a887c0953144090af30bb373a"),
    "render_pair.py": dict(bytes=7853, SHA256="ad8a7008539706bcd773311d8fef31b9e74b3ffd2d55a144b1923d300bec03f0"),
}
PLAN_LOCAL = REPO / "local-only/t07-style18-heldout-prelabel-v1/source-slots-to-review-v1.private.json"


def splice(text, old, new):
    contract.need(text.count(old) == 1, "SINGLE_ORIGINAL_INSERTION_POINT_REQUIRED")
    return text.replace(old, new, 1)


def target_sources():
    original = {name: (BASE / name).read_bytes() for name in PINS}
    contract.need({name: contract.pin(raw) for name, raw in original.items()} == PINS, "FROZEN_DEV12_SOURCE_CHANGED")
    worker = original["worker.py"].decode()
    worker = splice(worker, '    phase = "source"\n', '    phase = "heldout_source_plan"\n')
    worker = splice(worker, '    try:\n        verify_checkout',
                    '    try:\n        from heldout_contract import load_claim_plan\n'
                    '        load_claim_plan(claim)\n        phase = "source"\n        verify_checkout')
    worker = splice(worker, '        phase = "Style18_development_pair_render"',
                    '        phase = "Style18_heldout_pair_render"')
    worker = splice(worker, '        output = ROOT / "development"', '        output = ROOT / "heldout"')
    worker = splice(worker, '        result["development"] = render(model, torch, output, claim, check)\n'
                            '        need(result["development"]["passed"], "Development generation pair coverage differs")',
                            '        result["heldout"] = render(model, torch, output, claim, check)\n'
                            '        need(result["heldout"]["passed"], "Heldout generation pair coverage differs")')
    worker = splice(worker, '    print(json.dumps({name:result.get(name) for name in ("request","step","passed","phase","error_type","error")}))',
                    '    print(json.dumps({name:result.get(name) for name in ("request","step","passed","phase","error_type")}))')
    render = original["render_pair.py"].decode()
    render = splice(render, '"""Render Style18 dev pairs from the exact initial and step-25 LoRA values."""',
                    '"""Render frozen Style18 heldout pairs with the original strict two-arm loading."""')
    render = splice(render, 'import os\nimport time\n', 'from heldout_contract import execute, load_claim_plan\n')
    render = splice(render, 'PACKAGE = Path("/fusion-run")\n', '')
    render = splice(render, '    from PIL import Image,ImageChops,ImageStat\n', '')
    start = render.index('    plan_path=PACKAGE/')
    stop = render.index('    named=', start)
    render = render[:start] + '    samples=load_claim_plan(claim)\n' + render[stop:]
    stop = render.index('    pipe=model.pipe')
    render = render[:stop] + '    return execute(model,torch,output,samples,named,states,deadline,tensor_hash)\n'
    generated = {"worker.py": worker.encode(), "render_pair.py": render.encode(),
                 "heldout_contract.py": (HERE / "heldout_contract.py").read_bytes()}
    for name, raw in generated.items():
        compile(raw, name, "exec")
    a = next(n for n in ast.parse(original["render_pair.py"]).body if isinstance(n, ast.FunctionDef) and n.name == "render")
    b = next(n for n in ast.parse(generated["render_pair.py"]).body if isinstance(n, ast.FunctionDef) and n.name == "render")
    first = next(i for i,n in enumerate(a.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="named")
    change = next(i for i,n in enumerate(b.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="named")
    contract.need([ast.dump(n) for n in a.body[first:first+15]] == [ast.dump(n) for n in b.body[change:change+15]]
                  and ast.dump(a.body[3]) == ast.dump(b.body[2]), "ORIGINAL_CHECKPOINT_AND_NAMED_ADAPTER_GATE_MUST_STAY_IDENTICAL")
    return generated


def bind(args):
    by = args.claim.read_bytes()
    contract.need(len(by) <= 65536, "CLAIM_EXTENT")
    claim = json.loads(by)
    rows = contract.load_claim_plan(claim, args.plan)
    return claim, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="mode", required=True)
    build = cmds.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    for mode in ("preflight", "read"):
        cmd = cmds.add_parser(mode)
        cmd.add_argument("--claim", type=Path, required=True)
        cmd.add_argument("--plan", type=Path, default=PLAN_LOCAL)
        if mode == "read":
            cmd.add_argument("--heldout-output", type=Path, required=True)
            cmd.add_argument("--private-reconciled", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "build":
        generated = target_sources()
        args.output.mkdir(mode=0o700)
        for name, value in generated.items():
            with (args.output / name).open("xb") as stream:
                contract.durable(stream, value)
        manifest = dict(files={name: contract.pin(value) for name, value in generated.items()}, baseline_sources=PINS,
                        input_manifest=contract.PLAN, input_in_source_packet=False,
                        input_container_path=contract.PACKAGE_PLAN.as_posix(), artifact_root="/fusion-run/heldout",
                        planned_generations=32, planned_pairs=16)
        result = contract.save_exclusive(args.output / "STAGE_FILES.json", manifest)
    else:
        claim, rows = bind(args)
        result = dict(request=claim["request"], planned_generations=32, planned_pairs=len(rows), plan=contract.PLAN)
        if args.mode == "read":
            # The owner must already have established termination and process-exit custody.
            outcome = contract.reconcile(args.heldout_output, rows)
            result["private_reconciled"] = contract.save_exclusive(args.private_reconciled, outcome)
            result.update({k: outcome[k] for k in ("passed", "attempted", "completed", "failed", "missing",
                                                   "complete_same_noise_pairs", "malformed_ledger_rows")})
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
