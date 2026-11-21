"""Reserve the exact O01 placement before a single paired native compile."""

import json
from pathlib import PurePosixPath

from control_deep_rope_prefix_width_pair import (
    CLAIM as CLAIM_PATH, NODE, PAIR, PARENT, PLACEMENT, REPO, REQUEST, WORKER, WORKLOAD, ZONE, check_scope,
)
from run_deep_rope_prefix_width_compile_pair import read, require


HOST = "tools/control_deep_rope_prefix_width_pair.py"
MEMBER = "tools/read_training_slurm_membership.py"
DISPATCHER = "tools/dispatch_deep_rope_prefix_width_pair_once.py"
CLAIM = dict(path=CLAIM_PATH, bytes=1403, SHA256="f197aa27567f91bb08de947fcbc7c367da66f9ea059eddc918f9a963e24123b1")
HOST_SHA = "71830add9efdbf408ed13287b7feeea8c2ebefd805405e48e65b9b61fec1f70e"
WORKER_SHA = "3db2186173d84199cebde829110f88a6d75c819295c4ed91363c7ddc2a36d209"
MEMBER_SHA = "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3"
TEMPLATE = "tools/dispatch_deep_inverse_rope_compile_once.py"
TEMPLATE_SHA = "0ff53494e00e580822ae4d690a01df5074afd39306c838ec30c8593791ff0213"
IDENT = "deep-inverse-output-rope-prefix-width-pair-cpu-v1"
CASE = "O01.deep-inverse-output-rope-prefix-width-pair-compile"
INTENT = f"local-only/{PAIR}-original-dispatch/intent.json"


def backlog_identity():
    return dict(id=IDENT, case=CASE, kind="validation", state="blocked", cpu_prerequisite=True, cpu_request=REQUEST,
                placement={NODE: 0}, attempt_receipts=[INTENT], evidence=CLAIM)


def scope(repo):
    names = check_scope(repo, CLAIM["SHA256"])
    read(repo / TEMPLATE, 64 * 1024, TEMPLATE_SHA)
    claim = json.loads(read(repo / CLAIM_PATH, 8192, CLAIM["SHA256"]))
    require(claim["host_files"][HOST] == HOST_SHA and claim["host_files"][WORKER] == WORKER_SHA
            and claim["host_files"][MEMBER] == MEMBER_SHA, "frozen paired entry differs")
    return claim, names | {DISPATCHER, TEMPLATE, "tools/collect_training80_dispatch_snapshot.py",
                          "tools/report_training80_backlog.py", "tools/sample_training80_busy.py", "tools/tuning_queue.py"}


def original_argv(remote_repository, *, reconcile=False):
    root = PurePosixPath(remote_repository)
    require(root.is_relative_to(PROJECT) and root.as_posix() == remote_repository and ".." not in root.parts
            and all(part.isascii() and all(ch.isalnum() or ch in "-_." for ch in part) for part in root.parts[1:]),
            "expected a canonical target checkout within the project")
    bootstrap = (
        "import hashlib,runpy,sys\n"
        "from pathlib import Path\n"
        "paths=[Path(value) for value in sys.argv[1:4]]\n"
        "for path,pin in zip(paths,sys.argv[4:7],strict=True):\n"
        "    if path.resolve(strict=True)!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:\n"
        "        raise ValueError('frozen paired controller source differs')\n"
        "sys.path.append(str(paths[0].parent))\n"
        "sys.argv=[str(paths[0]),'--claim-sha256',sys.argv[7],*sys.argv[8:]]\n"
        "runpy.run_path(str(paths[0]),run_name='__main__')\n"
    )
    flags = ("--reconcile-existing",) if reconcile else ()
    return ("/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-u", "-c", bootstrap,
            str(root / HOST), str(root / MEMBER), str(root / WORKER), HOST_SHA, MEMBER_SHA, WORKER_SHA, CLAIM["SHA256"],
            *flags)


_FROZEN = {"__name__": "frozen_deep_pair_dispatch", "__file__": str(REPO / TEMPLATE)}
exec(compile(read(REPO / TEMPLATE, 64 * 1024, TEMPLATE_SHA), _FROZEN["__file__"], "exec"), _FROZEN)
PROJECT = _FROZEN["PROJECT"]
BACKLOG, QUEUE_DB = _FROZEN["BACKLOG"], _FROZEN["QUEUE_DB"]
TARGET = PROJECT / "results" / (PAIR + "-controller")
_FROZEN.update(
    HOST=HOST, HOST_FILES={HOST: HOST_SHA}, LABEL=PAIR, REQUEST=REQUEST, PARENT=PARENT, NODE=NODE, ZONE=ZONE,
    PLACEMENT=PLACEMENT, IDENT=IDENT, CASE=CASE, INTENT=INTENT, CLAIM=CLAIM, MANIFEST={"SHA256": WORKLOAD},
    AUDIT={"SHA256": HOST_SHA}, TARGET=TARGET, scope=scope, backlog_identity=backlog_identity, original_argv=original_argv,
)
dispatch = _FROZEN["dispatch"]


if __name__ == "__main__":
    print(json.dumps(backlog_identity(), indent=2, allow_nan=False))
