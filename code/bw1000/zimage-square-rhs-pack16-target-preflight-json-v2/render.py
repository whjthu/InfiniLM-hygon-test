"""Render a diagnostic preflight only from a separately admitted live successor."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
BASE = HERE.with_name("zimage-square-rhs-pack16-target-preflight-v1") / "precheck.py"
BASE_SHA = "2d6152e99c3401dbd05c2c63d64212807df855e98f01bfe030af08e6e51928a0"
OLD_STEP = "1103606.2480"
OLD_REQUEST = "30f9d5dd39bc49c1b8d52312ba2951d6"
OLD_STANDBY = "9454531ca7ef48e9a18f73d5ed411683"
GPU_REQUEST = "a31421758fc74d319923dbce7045cc12"
OLD_OBSERVED = 1795514898.7431285


def pin(data):
    return sha256(data).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("exact predecessor anchor differs")
    return text.replace(old, new, 1)


def binding(raw, digest):
    if not 0 < len(raw) <= 8192 or pin(raw) != digest:
        raise ValueError("live successor binding SHA differs")
    b = json.loads(raw)
    step, target, request = (b[key] for key in ("standby_step", "standby_request", "precheck_request"))
    label = "training80-dcu-standby-" + target[:12] + "-f12r2n10"
    if (b["schema"] != "bw1000.o01.pack16-preflight-live-successor.v2" or b["node"] != "f12r2n10"
            or b["parent"] != "1103606" or b["standby_label"] != label
            or re.fullmatch(r"1103606\.[0-9]+", step) is None or step in (OLD_STEP, "1103606.2569")
            or any(re.fullmatch("[0-9a-f]{32}", value) is None for value in (target, request))
            or target == OLD_STANDBY or request in {OLD_REQUEST, GPU_REQUEST, target}
            or any(re.fullmatch("[0-9a-f]{64}", b[field]) is None
                   for field in ("worker_SHA256", "original_boot_SHA256", "live_evidence_SHA256"))
            or b["state"] != "RUNNING" or b["actual_DCU"] != 8 or b["counter_growth_DCU"] != 8
            or not isinstance(b["observed_epoch"], (float, int)) or not math.isfinite(b["observed_epoch"])
            or b["observed_epoch"] <= OLD_OBSERVED):
        raise ValueError("distinct independently observed live successor is required")
    return b


def overlay(original, diagnostic, b, binding_sha):
    if pin(original) != BASE_SHA:
        raise ValueError("original terminal worker source SHA differs")
    text = original.decode()
    text = replace(text, '"' + OLD_STEP + '"', repr(b["standby_step"]))
    text = replace(text, '"' + OLD_STANDBY + '"', repr(b["standby_request"]))
    text = replace(text, '"' + OLD_REQUEST + '"', repr(b["precheck_request"]))
    text = replace(text, '"O01-pack16-deps-"', '"O01-pack16-deps-json-"')
    anchor = "\n\nDIAG_PHASES = "
    if diagnostic.decode().count(anchor) != 1:
        raise ValueError("diagnostic source boundary differs")
    helpers = anchor + diagnostic.decode().split(anchor, 1)[1]
    text = replace(text, 'import sys\n', 'import sys\n' + helpers + "\n")
    text = replace(text, 'OUT = ROOT / "results" / LABEL',
                   'OUT = ROOT / "results" / LABEL\n'
                   f'ADMISSION_SHA = {binding_sha!r}\n'
                   f'LIVE_EVIDENCE_SHA = {b["live_evidence_SHA256"]!r}\n'
                   f'WORKER_SHA = {b["worker_SHA256"]!r}\nBOOT_SHA = {b["original_boot_SHA256"]!r}')
    text = replace(text, "def worker_identity():", "def worker_identity(phase):")
    text = replace(text, "    worker = json.loads(raw)", """    worker = decode(phase, raw)
    if sha256(raw).hexdigest() != WORKER_SHA:
        raise PhaseFailure(phase, "PRECHECK_BOUND_SUCCESSOR_WORKER_SHA")""")
    text = replace(text, '[str(pid), PARENT, "2480"]', '[str(pid), PARENT, STANDBY_STEP.split(".")[1]]')
    text = replace(text, '("job_1103606", "step_2480")', '("job_1103606", "step_" + STANDBY_STEP.split(".")[1])')
    text = replace(text, '    return (pid, int(first[19]), pin(raw), sha256(read(BOOT, 128)).hexdigest(), sha256(cgroup).hexdigest())',
                   '''    boot = sha256(read(BOOT, 128)).hexdigest()
    if boot != BOOT_SHA:
        raise PhaseFailure(phase, "PRECHECK_BOUND_SUCCESSOR_BOOT_SHA")
    return (pid, int(first[19]), pin(raw), boot, sha256(cgroup).hexdigest())''')
    text = replace(text, "def runtime(path):", "def runtime(path, phase):")
    text = replace(text, 'manifest = json.loads(read(path / "SCOPED_FILES.json", 8192, RUNTIME_SHA))',
                   'manifest = decode(phase, read(path / "SCOPED_FILES.json", 8192, RUNTIME_SHA))')
    text = replace(text, "def inside():", "def inside(source, sha):\n    read(source, 32768, sha)")
    text = replace(text, 'runtime(Path("/pair-package"))', 'runtime(Path("/pair-package"), "runtime_inside")')
    text = replace(text, '"torch": torch_metadata(), "GPU_context_initializations": 0, "hipprof_invocations": 0}',
                   '"torch": torch_metadata(), "GPU_context_initializations": 0, "hipprof_invocations": 0, "target_GPU_GO": False}')
    text = replace(text, "def container_argv(source):", "def container_argv(source, sha):")
    text = replace(text, 'source /opt/dtk/env.sh; ', 'source /opt/dtk/env.sh >&2; ')
    text = replace(text, "'exec python3 -B /precheck.py inside')",
                   """'exec python3 -B /precheck.py inside "$@"')""")
    text = replace(text, '"/bin/bash", "--noprofile", "--norc", "-c", shell]',
                   '"/bin/bash", "--noprofile", "--norc", "-c", shell, "precheck", "--source-sha256", sha]')
    text = replace(text, "    worker = worker_identity()", '    worker = worker_identity("standby_before")')
    text = replace(text, '    checked = runtime(STAGE / "runtime")', '    checked = runtime(STAGE / "runtime", "runtime_host")')
    text = replace(text, """    got = subprocess.run(container_argv(source), env=env, cwd="/", stdin=subprocess.DEVNULL,
                         capture_output=True, timeout=32, check=False)
    require(got.returncode == 0 and 0 < len(got.stdout) <= 8192, "PRECHECK_CONTAINER_FAILED")
    result = json.loads(got.stdout)""", "    result = run_container(container_argv(source, sha), env, OUT, REQUEST, step)")
    text = replace(text, 'worker_identity() == worker', 'worker_identity("standby_after") == worker')
    text = replace(text, '"Slurm_GPU_resources": 0, "target_GPU_GO": False}',
                   '"Slurm_GPU_resources": 0, "target_GPU_GO": False,\n'
                   '            "binding_SHA256": ADMISSION_SHA, "live_evidence_SHA256": LIVE_EVIDENCE_SHA}')
    text = replace(text, 'parser.add_argument("--source-sha256")', 'parser.add_argument("--source-sha256", required=True)')
    text = replace(text, 'else inside()', 'else inside(source, args.source_sha256)')
    start = text.index("    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as error:")
    stop = text.index("    sys.stdout.buffer.write(encode(result))", start)
    text = text[:start] + """    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as error:
        detail = str(error)
        step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
        result = {"request": REQUEST, "step": step if re.fullmatch(r"1103606\\.[0-9]+", step) else None,
                  "precheck_passed": False, "failure_type": type(error).__name__, "target_GPU_GO": False,
                  "gate": detail if re.fullmatch("[A-Z][A-Z0-9_]+", detail) else None}
        if isinstance(error, PhaseFailure):
            result.update(error.details)
        sys.stdout.buffer.write(encode(result))
        return 1
""" + text[stop:]
    compiled = text.encode()
    compile(compiled, "precheck-v2.py", "exec")
    return compiled


def build(raw, expected):
    b = binding(raw, expected)
    code = overlay(BASE.read_bytes(), (HERE / "diagnostics.py").read_bytes(), b, expected)
    return code, {"request": b["precheck_request"], "source_SHA256": pin(code), "bytes": len(code),
                  "original_preflight_source_SHA256": BASE_SHA, "live_binding_SHA256": expected,
                  "replay_allowed": False, "render_does_not_authorize_launch": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.binding.open("rb") as stream:
        raw = stream.read(8193)
    code, receipt = build(raw, args.binding_sha256)
    with args.out.open("xb") as output:
        output.write(code)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
