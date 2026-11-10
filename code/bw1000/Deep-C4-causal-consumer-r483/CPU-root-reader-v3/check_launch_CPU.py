"""Validate the original CPU-v2 reader with local fixture bytes."""
import ast
import json
from pathlib import Path, PurePosixPath
import read_original_build as ns

base = Path(__file__).resolve().parents[1] / "CPU-v2"
epoch, end = 1793952000.25, 1793952300.25
# Evaluate only the argv list from the frozen launcher's source. No launcher is invoked.
launcher = ast.parse((base / "launch_CPU.py").read_bytes())
argv_expr = next(
    node.value for node in launcher.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    for node in node.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
)
argv = eval(compile(ast.Expression(argv_expr), "<frozen launch argv>", "eval"), {
    "epoch": epoch, "deadline": end, "LABEL": ns.LABEL, "ch": ns.CLAIM,
    "R": PurePosixPath(ns.HOST_ROOT),
    "D": PurePosixPath(ns.HOST_ROOT + ns.PACKAGE.removeprefix("/refactor")),
})
intent = dict(publication=ns.PUBLICATION, source_manifest_SHA256=ns.SOURCE_MANIFEST,
              claim_SHA256=ns.CLAIM, once=True, GPU=False, Native_dlopen=False,
              HOST="recorded-host", epoch=epoch, deadline=end, argv=argv)
launch = dict(HOST=intent["HOST"], epoch=epoch, deadline=end, GPU=False,
              owned_srun_PID=123, PGID=123, startticks="456", observed_at_epoch=epoch+1,
              intent=f"{ns.HOST_ROOT}/results/{ns.LABEL}-intent.json")
job = "910407.98765"
accounting = dict(rc=0, columns=ns.COLUMNS, stdout=f"{job}|{ns.LABEL}|COMPLETED|0:0|17")
def unexpected_read(path):
    raise AssertionError(f"launch gate attempted to read {path}")

accepted = ns.verify_terminal(intent, launch, accounting, job)
assert accepted["can_inspect_successful_build"]
mutations = {
    "extra_resource_overrides": argv[:14] + ["--cpus-per-task=8", "--gres=gpu:1"] + argv[14:],
    "different_cpu_entry_point": argv[:-3] + ["bash", "/tmp/different.sh", "probe"],
    "omitted_exact_and_overlap": [arg for arg in argv if arg not in ("--exact", "--overlap")],
    "duplicate_epoch": argv[:-3] + ["TUNING_CPU_EPOCH=" + repr(epoch + 1)] + argv[-3:],
    "reordered_epoch_and_deadline": argv[:19] + argv[19:21][::-1] + argv[21:],
}
rejected = []
for name, mutated_argv in mutations.items():
    try:
        ns.read_build(dict(intent, argv=mutated_argv), launch, accounting, job, None, unexpected_read, unexpected_read)
    except ValueError as error:
        assert "ordered launcher argv" in str(error)
        rejected.append(name)
    else:
        raise AssertionError(f"accepted changed launch argv: {name}")
try:
    ns.read_build(intent, dict(launch, intent="/tmp/another-intent.json"),
                  accounting, job, None, unexpected_read, unexpected_read)
except ValueError as error:
    assert "launch intent path" in str(error)
else:
    raise AssertionError("accepted different launch intent path")
print(json.dumps(dict(frozen_launch_argv_accepted=True, rejected_argv_mutations=rejected,
                      rejected_intent_path=True, artifact_reads_on_bad_launch=0)))
