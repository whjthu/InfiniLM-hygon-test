"""Bind the independently compiled route-counter library to the real TP8 owner."""

import hashlib
import json
from pathlib import Path

from render import checked_replace, render_model_owner


REPO = Path(__file__).resolve().parents[4]
SCOPE = Path(__file__).resolve().parent
PRIOR = SCOPE.parent / "inverse-output-rope-width-model-v1"
LABEL = "deep-rope-tp8-route-attribution-v1"
REQUEST = "1aa5f68429f34171b34d027b8cc5062b"
CPU_BUILD_SHA256 = "93f30fa73ddcf0c7e72de8ce0eb76b2191865c885be233a42c706339aaa1e2d2"
LIBRARY = "/refactor/results/deep-rope-route-tu15-cpu-v3/artifacts/libDeep_inverse_rope_width_routes.so"
ORIGINAL_CLAIM_SHA256 = "2ab32b5cd4b34bf2c2e168461e127ff8f18c7bedd0788b614b91ff2618c05269"
ORIGINAL_LAUNCH_SHA256 = "5aa8cd1b84faee605208827cb3b2bc036d3493881fe073bc3b0a01c881846187"
OBSERVER_SHA256 = "dd08bd50072c5ab56b15300477bb0dae94819cde558306f3eac7cb723ab989b9"


def pin(raw):
    return {"bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}


def read(path, expected):
    raw = path.read_bytes()
    if pin(raw)["SHA256"] != expected:
        raise ValueError(f"Frozen source differs: {path.name}")
    return raw


def materials():
    claim = json.loads(read(PRIOR / "GPU_CLAIM_V2.json", ORIGINAL_CLAIM_SHA256))
    source = {
        name: read(REPO / item["path"], item["SHA256"])
        for name, item in claim["materials"].items()
    }
    receipt = json.loads(read(
        REPO / "results/bw1000/2026-11-23/Deep-rope-route-counter-original1429-target-build-terminal.json",
        CPU_BUILD_SHA256,
    ))
    if receipt["library_container_path"] != LIBRARY or not receipt["original"]["passed"]:
        raise ValueError("Original CPU build is unqualified")
    owner = render_model_owner(source["model_owner.py"], LIBRARY).decode()
    owner = checked_replace(
        owner, "            route_after = routes.snapshot()\n",
        "            route_after = routes.snapshot()\n"
        '            emit("inverse_rope_route_snapshots", before=route_before, after=route_after)\n',
    )
    source["model_owner.py"] = owner.encode()
    source["observe_routes.py"] = read(SCOPE / "observe_routes.py", OBSERVER_SHA256)
    source["inside.sh"] = checked_replace(
        source["inside.sh"].decode(),
        '    export INFINILM_DEEP_INVERSE_OUTPUT_ROPE="$1"',
        '    export INFINILM_DEEP_INVERSE_OUTPUT_ROPE="$1" INFINILM_DEEP_INVERSE_ROPE_ROUTE_COUNTS=1',
    ).encode()
    launch = read(PRIOR / "model_launch.py", ORIGINAL_LAUNCH_SHA256).decode()
    start = launch.index("    aborted = PROJECT / ")
    stop = launch.index("    epoch = time.time()", start)
    launch = checked_replace(launch, launch[start:stop], "    source.admit(labels)\n")
    launch = checked_replace(launch, 'LABEL = "deep-rope-real-model-tp8-diagnostic-v2"', f"LABEL = {LABEL!r}")
    launch = checked_replace(launch, 'ABORTED_SCOPE = "deep-rope-real-model-tp8-diagnostic-v1"\n', "")
    launch = checked_replace(launch, '"split_admission.py", "model_launch.py", "preflight_once.py"}',
                             '"split_admission.py", "model_launch.py", "preflight_once.py", "observe_routes.py"}')
    launch = checked_replace(
        launch, "    scope = PROJECT / \"results\" / LABEL\n",
        "    native = PROJECT / claim[\"instrumented_library\"].removeprefix(\"/refactor/\")\n"
        "    require(pin(native.read_bytes()) == claim[\"assets\"][\"challenger\"],\n"
        "            \"Instrumented library bytes differ\")\n"
        "    scope = PROJECT / \"results\" / LABEL\n",
    )
    source["model_launch.py"] = launch.encode()
    for name in ("model_owner.py", "model_launch.py", "observe_routes.py"):
        compile(source[name], name, "exec")
    claim.update(label=LABEL, request=REQUEST, instrumented_library=LIBRARY,
                 previous_native_build={"step": "1103606.1429", "receipt_SHA256": CPU_BUILD_SHA256},
                 scope="Real TP8 route attribution in fixed legacy/pack32/pack64 arms. Capture positive and "
                       "fallback route deltas. Node and arm remain confounded. No paired performance claim.")
    del claim["aborted_v1"], claim["aborted_v1_controller_log"]
    claim["assets"]["challenger"] = receipt["library"]
    claim["materials"] = {name: pin(raw) for name, raw in source.items()}
    source["CLAIM.json"] = (json.dumps(claim, sort_keys=True, allow_nan=False) + "\n").encode()
    return source
