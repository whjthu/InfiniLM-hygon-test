"""Link the two audited Kimi .637 score objects once in the original DTK container."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OLD_ROOT = ROOT.with_name("hygon-inference-20261013")
LABEL = "Kimi461-output-residual-wave-component-link-CPU-v1"
SCORE_LABEL = "Kimi461-output-residual-wave-score-gfx936-CPU-v1"
SCORE_STEP, SCORE_POST = "1103606.637", "1103606.639"
SCORE_PIN = "4f961a007757591d3c918f84e6f07753d6b1fbd6210a0f6f139093a21e810ee7"
MANIFEST = "d2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6"
PARENT, NODE, ZONE = "1103606", "f12r2n11", "O01"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
REQUEST = sha256(f"{LABEL}:{PARENT}:{NODE}:{SCORE_PIN}".encode()).hexdigest()[:32]
TASK = "coordination/tasks/kimi-r461-output-residual-wave-component-link-cpu-v1.json"
LINKER = "tools/link_kimi_r461_output_residual_wave.py"
MEMBERSHIP = "tools/read_training_slurm_membership.py"
LEDGER = "coordination/training/training80-active-node-task-map-v1.json"
ACCOUNT = "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py"
KIMI = "code/bw1000/Kimi-vocab-parallel-embedding-r461/"
OLD_CONTROL = KIMI + "output-residual-wave-host-compile-v1/host_once.py"
ENV = KIMI + "output-residual-stage-diagnostic-CPU-v1/CPU_inside.sh"
SCORE_RESULT = "results/bw1000/2026-11-22/Kimi461-original637-two-wave-objects-and-clean-post.json"
ENV_PREFIX = "456e3ab53c263e6d01857974c2c0abdef21a0ea1aea06f0dfa588b13fe9e11b3"
INPUT_CANDIDATES = (SCORE_LABEL, "Kimi461-output-residual-tile64-CPU-v1",
                    "Kimi461-output-residual-stage-diagnostic-CPU-v1",
                    "Kimi-vocab-parallel-embedding-r461-probe-link-CPU-v2")
INNER = "/tmp/kimi-wave-link"
INNER_GENERATED = INNER + "-generated"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, pin=None, limit=2**21):
    require(path.resolve(strict=True) == path, f"noncanonical input: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit and (pin is None or sha256(data).hexdigest() == pin),
            f"unexpected input bytes: {path.name}")
    return data


def helpers(path, data):
    scope = dict(__name__="pinned_kimi_link_helper", __file__=str(path))
    exec(compile(data, str(path), "exec"), scope)
    return scope


def score_accounting(source, original):
    row = source["row"]
    require(source["state"] == "COMPILE_VERIFIED" and source["independent_observation"] is True
            and source["observations"] == [row] and source["actual_DCU"] == 0
            and source["independent_post"]["independent_post"] == SCORE_POST
            and source["score_receipt_SHA256"] == SCORE_PIN and source["columns"] == original["COLUMNS"]
            and tuple(row[:4]) == (SCORE_STEP, SCORE_LABEL, "COMPLETED", "0:0")
            and row[5] == NODE and row[6] == "cpu=4,mem=8G,node=1",
            "wrong original Kimi compile evidence")
    return dict(columns=source["columns"], row=row)


def frozen(repo, task_sha):
    claim = json.loads(read(repo / TASK, task_sha))
    expected = dict(request=REQUEST, label=LABEL, placement=PLACEMENT, wall_seconds=300,
                    score_compile_step=SCORE_STEP, score_compile_post=SCORE_POST, score_receipt_SHA256=SCORE_PIN,
                    rendered_manifest_SHA256=MANIFEST, host_TUs=1, device_TUs=0,
                    CPU_GO=True, GPU_GO=False, Model_GO=False, replay_allowed=False)
    require(all(type(claim[key]) is type(value) and claim[key] == value for key, value in expected.items()),
            "Kimi link claim differs")
    manifest = HERE / "SOURCE_FILES.json"
    pins = json.loads(read(manifest, claim["source_manifest_SHA256"]))["files_SHA256"]
    own = Path(__file__).resolve().relative_to(repo).as_posix()
    require(set(pins) == {own, LINKER, MEMBERSHIP, LEDGER, ACCOUNT, OLD_CONTROL, ENV, SCORE_RESULT},
            "unexpected link manifest members")
    data = {name: read(repo / name, pin) for name, pin in pins.items()}
    original, membership, terminal = (helpers(repo / name, data[name]) for name in (OLD_CONTROL, MEMBERSHIP, ACCOUNT))
    ledger = json.loads(data[LEDGER])
    groups = {parent: {name: group["allocations"][parent] for name, group in ledger["groups"].items()}
              for parent in original["PARENTS"]}
    require(NODE in groups[PARENT][ZONE], "Kimi link node is outside original O01")
    upstream = score_accounting(json.loads(data[SCORE_RESULT]), original)
    prefix, marker, _ = data[ENV].partition(b"left=$(python")
    require(marker and sha256(prefix).hexdigest() == ENV_PREFIX, "original Kimi DTK environment differs")
    return data, original, membership, terminal, groups, upstream, prefix


def build_command(root, scope, generated, epoch):
    binds = [(root, "/refactor", "ro"), (root / "candidates", "/refactor/candidates", "rw"),
             (OLD_ROOT, "/old", "ro"), (Path("/opt/hyhal"), "/opt/hyhal", "ro"),
             (scope, INNER, "ro"), (generated, INNER_GENERATED, "ro")]
    binds.extend((root / "candidates" / name, "/refactor/candidates/" + name, "ro") for name in INPUT_CANDIDATES)
    container = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--pwd", "/",
                 "--home", str(OLD_ROOT / "build-deps/root") + ":/root"]
    for source, target, mode in binds:
        container.extend(("--bind", f"{source}:{target}:{mode}"))
    return ["srun", f"--jobid={PARENT}", "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=4", "--cpu-bind=threads", "--mem=8192M", "--gres=none", f"--nodelist={NODE}",
            "--time=00:05:00", f"--job-name={LABEL}", *container, str(root / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", INNER + "/inside.sh", repr(epoch), repr(epoch+300)]


def inside_script(prefix):
    run_linker = ("import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                  "runpy.run_path(sys.argv.pop(1), run_name='__main__')")
    worker = ["python", "-I", "-S", "-B", "-u", "-c", run_linker, INNER, INNER + "/linker.py",
              "--generated-source", INNER_GENERATED, "--score-receipt-sha256", SCORE_PIN,
              "--score-accounting", INNER + "/score-accounting.json"]
    # The independently captured .637 objects are mounted read-only. Retain the .335 SDK and loader paths.
    return prefix + (f'export TUNING_CPU_PARENT={PARENT} TUNING_CPU_NODE={NODE}\n'
                     'export TUNING_CPU_EPOCH="$1" TUNING_CPU_DEADLINE="$2"\n'
                     'export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1\n'
                     'unset GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH\n'
                     'budget=$(("${TUNING_CPU_DEADLINE%.*}" - $(/usr/bin/date +%s) - 12))\n'
                     '((budget > 1)) || exit 124\n'
                     'exec /usr/bin/timeout --kill-after=5s "${budget}s" ' + shlex.join(worker) + "\n").encode()


def link_accounting(original, timeout):
    columns = original["COLUMNS"]
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", ",".join(original["PARENTS"]),
                                   "-S", "2026-11-20T00:00:00", "-o", ",".join(columns)], timeout=timeout)
    require(len(raw) <= 2**21, "oversized Kimi link accounting")
    rows = [line.split("|") for line in raw.decode().splitlines() if line]
    require(all(len(row) == len(columns) for row in rows), "unexpected Kimi link accounting")
    matches = [row for row in rows if row[1] == LABEL]
    require(len(matches) <= 1 and all(re.fullmatch(PARENT+r"\.\d+", row[0]) and row[5] == NODE for row in matches),
            "duplicate or misplaced Kimi link label")
    return matches[0] if matches else None


def result(row, epoch, output, original, membership, terminal):
    if not terminal["is_terminal"](row):
        return dict(state="UNKNOWN")
    state = dict(state="TERMINAL_UNVERIFIED", columns=original["COLUMNS"], row=row)
    if row[2:4] != ["COMPLETED", "0:0"]:
        return state
    try:
        fields = membership["completed_numbered_step"](state["columns"], row, row[0], scope="Kimi wave link")
        resources = dict(field.split("=", 1) for field in fields["AllocTRES"].split(","))
        raw = read(output / "CPU-result.json")
        receipt = json.loads(raw)
        identity = dict(step=row[0], node=NODE, epoch=epoch, deadline=epoch+300, score_compile_step=SCORE_STEP,
                        score_receipt_SHA256=SCORE_PIN, generated_manifest_SHA256=MANIFEST, passed=True,
                        completed=True, device_calls=0, weights_read=0)
        require(resources.get("cpu") == "4" and resources.get("node") == "1"
                and resources.get("mem") in {"8G", "8192M"} and membership["allocated_dcu"](fields["AllocTRES"]) == 0
                and fields["ElapsedRaw"].isdecimal() and int(fields["ElapsedRaw"]) <= 300
                and all(type(receipt[key]) is type(value) and receipt[key] == value for key, value in identity.items())
                and set(receipt["arms"]) == {"full-tree", "wave32"}, "unexpected original link result")
        first, second = (receipt["arms"][name] for name in ("full-tree", "wave32"))
        require(first["SHA256"] != second["SHA256"] and first["exports"] == second["exports"]
                and first["providers"] == second["providers"], "linked provider or export identities differ")
        for arm in ("full-tree", "wave32"):
            lib = output / (arm + ".so")
            require(receipt["arms"][arm]["path"] == "/refactor/candidates/" + LABEL + "/" + lib.name,
                    "linked library path differs")
            read(lib, receipt["arms"][arm]["SHA256"], 32*1024**2)
        state.update(state="LINK_VERIFIED", link_receipt_SHA256=sha256(raw).hexdigest())
    except (OSError, KeyError, TypeError, ValueError) as error:
        state["error"] = str(error)[:200]
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    require(REPO.is_relative_to(ROOT) and ROOT.resolve(strict=True) == ROOT, "expected original project checkout")
    data, original, membership, terminal, groups, upstream, prefix = frozen(REPO, args.claim_sha256)
    scope = ROOT / "results" / (LABEL + "-controller")
    generated = ROOT / "results" / (SCORE_LABEL + "-controller") / "generated"
    output = ROOT / "candidates" / LABEL
    save_once = original["save_once"]
    query = lambda timeout: link_accounting(original, timeout)
    if args.reconcile:
        raw = read(scope / "intent.json")
        intent, pin = json.loads(raw), sha256(raw).hexdigest()
        epoch = intent["epoch"]
        attempt = json.loads(read(scope / "exec-attempt.json"))
        require(intent["request"] == REQUEST and intent["claim_SHA256"] == args.claim_sha256
                and intent["label"] == LABEL and intent["placement"] == PLACEMENT
                and intent["score_receipt_SHA256"] == SCORE_PIN and intent["replay_allowed"] is False
                and type(epoch) is float and math.isfinite(epoch) and 0 < epoch < epoch+300
                and intent["deadline"] == epoch+300
                and attempt["request"] == REQUEST and attempt["intent_SHA256"] == pin
                and attempt["argv"] == build_command(ROOT, scope, generated, epoch)
                and attempt["replay_allowed"] is False
                and epoch <= attempt["attempted_epoch"] < epoch+300, "unbound original Kimi link attempt")
        row = query(8)
        observed = result(row, epoch, output, original, membership, terminal)
        observed.update(request=REQUEST, intent_SHA256=pin, independent_observation=True, observations=[row])
    else:
        require(not any(path.exists() or path.is_symlink() for path in (scope, output)), "original namespace exists")
        require(query(8) is None, "original Kimi link step already exists")
        original["generated_payload"](generated)
        read(ROOT / "candidates" / SCORE_LABEL / "CPU-result.json", SCORE_PIN)
        expires = original["parent_expiry"](membership)
        scope.mkdir(mode=0o700)
        for name, content in (("linker.py", data[LINKER]), ("read_training_slurm_membership.py", data[MEMBERSHIP]),
                              ("inside.sh", inside_script(prefix))):
            with (scope / name).open("xb") as stream:
                stream.write(content)
        save_once(scope / "score-accounting.json", upstream)
        slot = original["placement"](membership["snapshot"](groups, owner_uid=os.getuid()), time.time())
        epoch, started = time.time(), time.monotonic()
        require(expires >= epoch+300, "allocation expires before the Kimi link deadline")
        pin = save_once(scope / "intent.json", dict(request=REQUEST, label=LABEL, claim_SHA256=args.claim_sha256,
                        placement=PLACEMENT, epoch=epoch, deadline=epoch+300, score_receipt_SHA256=SCORE_PIN,
                        preflight=slot, replay_allowed=False))
        observed = dict(request=REQUEST, intent_SHA256=pin, state="PRELAUNCH_REFUSED",
                        exec_request_attempted=False, actual_DCU=0)

        def left():
            return min(epoch+300-time.time(), 300-(time.monotonic()-started))

        try:
            require(query(8) is None and 0 <= time.time()-slot["observed_epoch"] <= 10,
                    "original label or membership changed before the link")
            argv = build_command(ROOT, scope, generated, epoch)
            with (scope / "srun.log").open("xb") as log:
                save_once(scope / "exec-attempt.json", dict(request=REQUEST, intent_SHA256=pin, argv=argv,
                          attempted_epoch=time.time(), replay_allowed=False))
                observed.update(state="UNKNOWN", exec_request_attempted=True)
                try:
                    observed["client_rc"] = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, timeout=left()-25, check=False).returncode
                except subprocess.TimeoutExpired:
                    observed["client_timeout"] = True

            def poll():
                return query(min(8, max(1, left()-1)))

            seen = terminal["wait_terminal"](poll, time.monotonic()+min(18, left()-2), max_observations=6)
            observed.update(result(seen["row"], epoch, output, original, membership, terminal),
                            observations=seen["observations"])
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            observed["error"] = f"{type(error).__name__}: {str(error)[:180]}"
        finally:
            save_once(scope / "accounting-result.json", observed)
    print(json.dumps(observed, allow_nan=False), flush=True)
    return int(observed["state"] != "LINK_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
