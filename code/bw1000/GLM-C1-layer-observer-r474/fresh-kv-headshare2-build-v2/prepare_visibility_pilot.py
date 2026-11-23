"""Use the qualified GLM event runner's device bindings for a new pilot."""

import json

from prepare_mounted_pilot import REQUEST as PRIOR, build_inputs as previous_inputs, digest, replace


REQUEST = "0b4f06465f1b46839169fdebd964845d"
FAILED_SHA256 = "4bd47cb4daacdac431a74bcf2ab417244ec1cb68ac3ca95a57fba735be88f4e0"


def build_inputs(repo):
    failed = (repo / "results/bw1000/2026-11-23/GLM-headshare-pilot-original1453-device-visibility-terminal.json").read_bytes()
    prior = json.loads(failed)
    if (digest(failed) != FAILED_SHA256 or prior["request"] != PRIOR or not prior["post_clean"]
            or prior["pack_kernel_launches_confirmed"] != 0):
        raise ValueError("Previous GPU visibility terminal is missing")
    files = previous_inputs(repo)
    files.pop("SOURCE_FILES.json")
    env = (repo / "code/bw1000/GLM-C1-layer-observer-r474/depth-stratified-observer-CPU-v1/ENV.sh").read_bytes()
    if digest(env) != "147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9":
        raise ValueError("Original qualified device environment differs")
    files["device_ENV.sh"] = env
    launcher = replace(files["launch_plan.py"], f"PILOT_REQUEST = {PRIOR!r}\n".encode(),
                       f"PILOT_REQUEST = {REQUEST!r}\n".encode())
    launcher = replace(launcher, b"                 SANDBOX,", b"                 '--bind','/opt/hyhal:/opt/hyhal:ro', SANDBOX,")
    files["launch_plan.py"] = replace(
        launcher,
        b"        'set -euo pipefail; source /opt/dtk/env.sh; node=$(hostname -s); '\n",
        b"        'set -euo pipefail; declare -A inherited_masks; '\n"
        b"        'for mask in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do '\n"
        b"        'if [[ -v $mask ]]; then inherited_masks[$mask]=${!mask}; fi; done; '\n"
        b"        'source /headshare/device_ENV.sh; '\n"
        b"        'for mask in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do '\n"
        b"        'if [[ -v inherited_masks[$mask] ]]; then export \"$mask=${inherited_masks[$mask]}\"; '\n"
        b"        'else unset \"$mask\"; fi; done; node=$(hostname -s); '\n",
    )
    compile(files["launch_plan.py"], "launch_plan.py", "exec")
    files["SOURCE_FILES.json"] = (json.dumps({
        "files_SHA256": {name: digest(raw) for name, raw in files.items()},
        "pilot_request": REQUEST,
        "previous_terminal_SHA256": FAILED_SHA256,
    }, sort_keys=True, indent=2) + "\n").encode()
    return files
