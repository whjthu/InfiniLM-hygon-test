"""Link the original Kimi wave32 object once using the already-built HOST object."""

from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import socket
import stat
import subprocess
import sys
import time


LABEL = "Kimi461-original678-wave32-link-only-rocm-CPU-v1"
REQUEST = "c55ff6edbcef4919a56ba16a78f379b9"
PARENT, NODE = "1103606", "f12r2n13"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
CANDIDATES = Path("/refactor/candidates")
OUTPUT = Path("/refactor/results") / LABEL
HOST = CANDIDATES / "Kimi461-output-residual-wave-component-link-CPU-v2"
SCORES = CANDIDATES / "Kimi461-output-residual-wave-score-gfx936-CPU-v1"
STAGE = CANDIDATES / "Kimi461-output-residual-stage-diagnostic-CPU-v1"
REPAIR = Path("/refactor/results/Kimi461-original678-wave-passive-rocm-CPU-v1/repair-result.json")
REPAIR_SHA = "a6230f278fa7ef63831c3fc9425307adee3ead9e41e1ae9aa3b00c5b5ea49cd2"
REPAIR_SOURCE_SHA = "ddaeee957e268f9e0b3391292ec2812c492bb2d5f6726659a819601bff8df5ad"
COMPILER_RECEIPT = CANDIDATES / "Kimi-vocab-parallel-embedding-r461-probe-link-CPU-v2/CPU-result.json"
COMPILER_RECEIPT_SHA = "982f780949434bbee15a336ea1e67e78d8482b77913a55b2f9987b959fe08217"
SCORE_RECEIPT_SHA = "4f961a007757591d3c918f84e6f07753d6b1fbd6210a0f6f139093a21e810ee7"
RENDERED_SHA = "d2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6"
HOST_SHA = "c0adf65d44801afb1abbd66531c65dfcfad08330da1ef5ac5e6f0c645f027005"
HOST_DEP_SHA = "943e7f1f95903b3643c722d265268681716921f3185ea34b599bccf1a7227e9f"
HOST_SOURCES = {
    "component.cpp": "39274b790a42e985deb7051eb8d5bbe16df7b3e6dc8558520dfc9aff0bd3f3c1",
    "residual_method.inc": "c5f56504c4e4cffb15943102e5887e96982b71a90ccf2fd73d8d3548d8bd61db",
    "output_residual_tiles.hpp": "d554d7e131e1eb5f6e2a39c1afd162acb1ba04124c67051991b91cd87116a9b2",
}
SCORE_SHA = {
    "full-tree": ("06f130a70b7b056fdb797bcaecfc08dd14966b9e9da1eaa4b6049b8f4f7d7030",
                  "a957f12349d0d12c37b2167d40d4f5af5323aca8ecd4c5cbed97d16dc0a8660e"),
    "wave32": ("ae313c4e68cfe0b598c474cc1279b698157ea16064076e34601b5c2679409509",
               "f337bf15f9ba29f75d8a9c1d5a77ed750c020b1ea0d29ed92100aad25fb496c9"),
}
VISIBLE = {"kimi_output_residual_component_r461", "kimi_output_residual_error_r461"}
HIDDEN = "kimi_output_residual_scores_v1_launch"
CAPTURE = "kimi_output_residual_scores_probe_r461"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def fixed(path, pin, limit=4 * 1024**2, size=None):
    need(path.resolve(strict=True) == path, f"source path differs: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(0 < len(data) <= limit and (size is None or len(data) == size)
         and sha256(data).hexdigest() == pin, f"source bytes differ: {path}")
    return data


def rocm_verifier():
    path = Path(__file__).resolve().with_name("rocm_verify.py")
    raw = fixed(path, REPAIR_SOURCE_SHA, limit=16 * 1024)
    spec = importlib.util.spec_from_file_location("kimi_wave32_exact_provider_verifier", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def link_argv(original, cpu330, cuda, output):
    compile_cmd, = (cmd for cmd in cpu330["commands"] if "-c" in cmd)
    link_cmd, = (cmd for cmd in cpu330["commands"] if "-shared" in cmd)
    flags = [flag for flag in link_cmd if flag.startswith(("-l", "-L"))]
    host = str(HOST / "component.o")
    source_score = str(SCORES / "full-tree/fused_scores.o")
    wave = str(SCORES / "wave32/fused_scores.o")
    expected = [compile_cmd[0], "-shared", "-pthread", "-Wl,-Bsymbolic", "-Wl,--no-undefined",
                host, source_score, str(cuda), *flags, "-o", str(HOST / "full-tree.so")]
    need(original == expected and PurePosixPath(expected[0]).is_absolute(), "previously executed .678 link recipe differs")
    return [*expected[:6], wave, *expected[7:-1], str(output)]


def validate_repair(repaired, source_sha):
    need(repaired["source_step"] == "1103606.678" and repaired["source_state"] == "FAILED"
         and repaired["observed_step"] == "1103606.748" and repaired["node"] == NODE
         and repaired["request"] == "f02bc5837890441fbe80a1fab68b598b"
         and all(repaired[key] is True for key in ("passed", "original_path_rule_pass", "original_digests_complete",
                                                   "original_dso_rechecked"))
         and repaired["original_dso_SHA256"] == source_sha and not repaired["byte_mismatches"]
         and repaired["loaded_count"] == repaired["provider_count_hashed"] == 68
         and len(set(repaired["aliases"].values())) == 68 and repaired["provider_bytes_read"] == 6644398160
         and repaired["passive_loader_calls"] == 1 and repaired["loader_exit_code"] == 0
         and all(repaired[key] == 0 for key in ("GPU_calls", "native_application_loads", "new_link_calls",
                                                "compiled_objects")),
         "original .748 passive qualification differs")


def dependency_names(raw):
    target, colon, dependencies = raw.decode("utf8").replace("\\\n", " ").partition(":")
    names = set(shlex.split(dependencies))
    need(colon and shlex.split(target) == [str(HOST / "component.o")]
         and names and all(name.startswith("/") for name in names), "original HOST depfile scope differs")
    return names


def exports(log):
    names = {part[-1] for line in log.splitlines() if len(part := line.split()) >= 3 and part[-2] == "T"}
    need(VISIBLE <= names and not ({HIDDEN, CAPTURE} & names), "wave link C ABI scope differs")
    return names


def hidden_score(log):
    return any(len(part := line.split()) >= 8 and part[-1] == HIDDEN and part[6] != "UND"
               and (part[4] == "LOCAL" or part[5] == "HIDDEN") for line in log.splitlines())


def main(epoch):
    wall, mono = time.time(), time.monotonic()
    deadline = epoch + 280
    job, step = (os.environ[key] for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    host = socket.gethostname().split(".")[0]
    need(job == PARENT and host == NODE and step.isdecimal() and len(os.sched_getaffinity(0)) == 4
         and all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and not any(os.environ.get(key) for key in ("SLURM_STEP_GPUS", "LD_PRELOAD", "LD_AUDIT"))
         and math.isfinite(epoch) and 0 < epoch <= wall < epoch + 240
         and OUTPUT.resolve(strict=True) == OUTPUT and OUTPUT.is_dir(), "link scope or lease differs")
    rocm = rocm_verifier()
    audit = rocm.audit_module()
    dso = OUTPUT / "wave32.so"
    report = dict(request=REQUEST, step=f"{job}.{step}", node=host, epoch=epoch, deadline=deadline,
                  original_678_state="FAILED", reused_passive_step="1103606.748", score_step="1103606.637",
                  arm="wave32", completed=False, passed=False, link_calls=0, compiled_HOST_objects=0,
                  compiled_HIP_objects=0, GPU_calls=0, native_application_loads=0,
                  dependency_bytes_read=0, provider_bytes_read=0, byte_mismatches=[],
                  original_path_rule_pass=False, original_digests_complete=False, commands=[])
    rocm.write_once(OUTPUT / "worker-attempt.json", {key: report[key] for key in ("request", "step", "epoch")})

    def seconds_left():
        return min(deadline - time.time(), deadline - wall - (time.monotonic() - mono))

    def digest(path, counter):
        digest = sha256()
        with Path(path).open("rb") as stream:
            before = os.fstat(stream.fileno())
            need(stat.S_ISREG(before.st_mode), "compiler or provider input is not a regular file")
            for raw in iter(lambda: stream.read(1024**2), b""):
                report[counter] += len(raw)
                need(seconds_left() > 5 and report[counter] <= 8 * 1024**3, "dependency digest budget exhausted")
                digest.update(raw)
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            need(all((row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns) == identity
                     for row in (os.fstat(stream.fileno()), Path(path).stat())), "dependency changed while reading")
        return digest.hexdigest()

    def run(phase, argv, cap, read_log=True):
        limit = min(cap, math.floor(seconds_left() - 9))
        need(limit > 0, "wave32 link command budget exhausted")
        report["commands"].append(dict(phase=phase, argv=argv))
        if phase == "wave32-link":
            report["link_calls"] += 1
        log = OUTPUT / (phase + ".log")
        with log.open("xb") as stream:
            result = subprocess.run(["/usr/bin/timeout", "--kill-after=3s", f"{limit}s", *argv],
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=limit + 5, check=False, env={**os.environ, "LC_ALL": "C"})
        report["commands"][-1]["exit_code"] = result.returncode
        need(result.returncode == 0, f"{phase} failed")
        if read_log:
            with log.open("rb") as stream:
                raw = stream.read(2 * 1024**2 + 1)
            need(0 < len(raw) <= 2 * 1024**2, f"{phase} output differs")
            report["commands"][-1]["log_SHA256"] = sha256(raw).hexdigest()
            return raw
        return b""

    try:
        need(not dso.exists() and not dso.is_symlink(), "wave32 output already exists")
        original_dso = HOST / "full-tree.so"
        rocm_fixed = audit.fixed
        original = json.loads(rocm_fixed(HOST / "CPU-result.json"))
        phases = ["HOST-fresh-M", "HOST-compile-once", "full-tree-link", "full-tree-exports",
                  "full-tree-symtab", "full-tree-passive"]
        need(original["step"] == "1103606.678" and original["completed"] is True and original["passed"] is False
             and original["reason"] == "component provider identity differs"
             and [item["phase"] for item in original["commands"]] == phases
             and original["score_receipt_SHA256"] == SCORE_RECEIPT_SHA and original["generated_manifest_SHA256"] == RENDERED_SHA,
             "original full-tree HOST/object relationship differs")
        source_dso_sha = audit.ORIGINAL[original_dso]
        rocm_fixed(original_dso)
        repaired = json.loads(fixed(REPAIR, REPAIR_SHA, size=9549))
        validate_repair(repaired, source_dso_sha)
        cpu330 = json.loads(rocm_fixed(CANDIDATES / "Kimi461-output-residual-tile64-CPU-v1/CPU-result.json"))
        cpu335 = json.loads(rocm_fixed(STAGE / "CPU-result.json"))
        need(cpu330["actual_CPU"] == "1103606.330" and cpu335["actual_CPU"] == "1103606.335"
             and all(row["completed"] is True and row["passed"] is True for row in (cpu330, cpu335)),
             "original HOST closure qualification differs")
        fixed(HOST / "component.o", HOST_SHA, size=792592)
        before = fixed(HOST / "component.before.d", HOST_DEP_SHA, size=74205)
        after = fixed(HOST / "component.d", HOST_DEP_SHA, size=74205)
        need(before == after, "original HOST fresh-M and actual MD differ")
        inventory = cpu335["dependency_SHA256"]
        previous = {name for name in inventory if "/output-residual-stage-diagnostic-CPU-v1/" in name}
        need({Path(name).name for name in previous} == {*HOST_SOURCES, "stage.hpp"} and len(previous) == 4,
             "original HOST producer dependency replacement differs")
        dependencies = {n: pin for n, pin in inventory.items() if n not in previous}
        dependencies.update({str(HOST / "host-source" / name): pin for name, pin in HOST_SOURCES.items()})
        need(dependencies.keys() == dependency_names(after), "original HOST dependencies differ")
        need(all(digest(name, "dependency_bytes_read") == pin for name, pin in sorted(dependencies.items())),
             "a current frozen HOST dependency differs")
        report["HOST_dependency_count"] = len(dependencies)
        score = json.loads(fixed(SCORES / "CPU-result.json", SCORE_RECEIPT_SHA))
        need(score["original_step"] == "1103606.637" and score["node"] == "f12r2n11"
             and score["completed"] is True and score["passed"] is True and set(score["arms"]) == set(SCORE_SHA)
             and score["manifest_SHA256"] == RENDERED_SHA and score["all_fresh_M_before_compile"] is True
             and score["actual_device_TUs"] == 2 and score["actual_GPU_steps"] == score["runtime_calls"] == 0
             and score["deadline"] == score["epoch"] + 300,
             "original score producer differs")
        for arm, (obj_sha, dep_sha) in SCORE_SHA.items():
            path = SCORES / arm / "fused_scores.o"
            row = score["arms"][arm]
            need(row["object"] == str(path) and row["object_SHA256"] == obj_sha
                 and row["fresh_M_SHA256"] == row["actual_MD_SHA256"] == dep_sha and row["exports"] == [HIDDEN],
                 f"original .637 {arm} receipt differs")
            fixed(path, obj_sha, limit=16 * 1024**2)
            fixed(path.with_name("fused_scores.before.d"), dep_sha)
            fixed(path.with_suffix(".d"), dep_sha)
        cmd = link_argv(original["commands"][2]["argv"], cpu330, audit.CUDART, dso)
        compiler = json.loads(fixed(COMPILER_RECEIPT, COMPILER_RECEIPT_SHA))
        known_compiler = compiler["compiler_current_container"]
        compiler_path = Path(cmd[0]).resolve(strict=True)
        need(compiler["actual_CPU"] == "910407.987" and compiler["passed"] is True and compiler["completed"] is True
             and str(compiler_path) == known_compiler["canonical"]
             and digest(compiler_path, "dependency_bytes_read") == known_compiler["SHA256"],
             "original HOST compiler provider differs")
        cuda = str(audit.CUDART.resolve(strict=True))
        need(digest(cuda, "dependency_bytes_read") == audit.CUDART_SHA, "original CUDA link input differs")
        original_exports = exports(run("original-exports", ["/usr/bin/nm", "-D", "--defined-only", str(original_dso)], 12).decode())
        run("wave32-link", cmd, 45, read_log=False)
        wave_exports = exports(run("wave32-exports", ["/usr/bin/nm", "-D", "--defined-only", str(dso)], 12).decode())
        symtab = run("wave32-symtab", ["/usr/bin/readelf", "-sW", str(dso)], 12).decode()
        need(wave_exports == original_exports and hidden_score(symtab), "wave32 exported or hidden ABI differs")
        passive = run("wave32-passive", ["/usr/bin/ldd", "-r", str(dso)], 25)
        qualified = {**cpu330["actual_passive_providers"], cuda: audit.CUDART_SHA}
        rocm.closure(audit, passive, qualified, set(cpu335["passive_providers"]), cuda,
                     lambda name: str(Path(name).resolve(strict=True)),
                     lambda name: digest(name, "provider_bytes_read"), report)
        need(set(report["aliases"].values()) == set(repaired["aliases"].values())
             and report["provider_bytes_read"] == repaired["provider_bytes_read"],
             "wave32 providers differ from the actual repaired full-tree closure")
        report["provider_count_hashed"] = report["loaded_count"]
        report["wave32_DSO"] = dict(path=str(dso), bytes=dso.stat().st_size,
                                    SHA256=digest(dso, "dependency_bytes_read"), exports=sorted(wave_exports))
        need(report["wave32_DSO"]["SHA256"] != source_dso_sha, "wave32 reused the baseline library")
        rocm_fixed(original_dso)
        fixed(HOST / "component.o", HOST_SHA, size=792592)
        fixed(SCORES / "wave32/fused_scores.o", SCORE_SHA["wave32"][0], limit=16 * 1024**2)
        report.update(original_full_tree_SHA256=source_dso_sha,
                      passive_full_tree_receipt_SHA256=REPAIR_SHA, passed=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report["reason"] = str(error)[:512]
    report.update(completed=True, elapsed_seconds=time.monotonic() - mono)
    rocm.write_once(OUTPUT / "link-result.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("expected launch epoch")
    raise SystemExit(main(float(sys.argv[1])))
