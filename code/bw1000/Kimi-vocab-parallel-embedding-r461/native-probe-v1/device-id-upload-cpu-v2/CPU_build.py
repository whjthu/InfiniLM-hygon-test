"""Compile only the audited Kimi worker with the corrected owner-side ID upload."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import time

D = Path(__file__).resolve().parent
BASE = Path("/refactor/candidates/Kimi-vocab-parallel-embedding-r461-probe-link-CPU-v2/CPU-result.json")
OUT = Path("/refactor/candidates/Kimi461-device-id-upload-CPU-v2")
HEADER = "probe_owned.hpp"


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def deps(path, target):
    first, sep, rest = path.read_text().replace("\\\n", " ").partition(": ")
    assert sep and shlex.split(first) == [str(target)]
    return set(shlex.split(rest))


def swap(argv, old, new):
    assert argv.count(str(old)) == 1
    argv[argv.index(str(old))] = str(new)


def main():
    claim_sha = os.environ["KIMI461_UPLOAD_CPU_CLAIM_SHA"]
    p = D / "CPU_RESOURCE_CLAIM.json"
    assert digest(p) == claim_sha
    q = json.loads(p.read_bytes())
    assert q["CPU_GO"] and q["once"] and not q["GPU_GO"] and not q["Model_GO"]
    assert (os.environ["SLURM_JOB_ID"], socket.gethostname().split(".")[0]) == (q["parent_job"], q["node"])
    assert all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    assert not os.environ.get("LD_PRELOAD") and not os.environ.get("LD_AUDIT")
    publication = os.environ["KIMI461_UPLOAD_CPU_PUBLICATION"]
    assert re.fullmatch("[0-9a-f]{40}", publication)
    for name, pin in q["files_SHA256"].items():
        assert digest(D / name) == pin
    assert digest(BASE) == q["base_receipt_sha256"]
    base = json.loads(BASE.read_bytes())
    assert base["passed"] and base["completed"] and base["actual_CPU"] == "910407.987"
    assert base["native_SHA256"] == q["parent_native_sha256"] == digest(base["native"])
    for path, pin in base["all_linked_object_SHA256"].items():
        assert digest(path) == pin
    assert len(base["all_linked_object_SHA256"]) == 34
    for item in base["actual_dependency_library_SHA256"].values():
        assert digest(item["path"]) == item["SHA256"]
    tool = base["compiler_current_container"]
    compiler = Path(shutil.which("g++")).resolve(strict=True)
    assert compiler == Path(tool["canonical"]) and digest(compiler) == tool["SHA256"]

    unit = base["original985_metadata"]["worker"]
    cmd = list(unit["actual_compile_argv"])
    old_obj = Path(cmd[cmd.index("-o")+1])
    old_md = Path(cmd[cmd.index("-MF")+1])
    assert str(old_obj) == q["original_worker_object"] and digest(old_obj) == q["original_worker_object_sha256"]
    assert digest(old_md) == unit["original_MD"][".d"]["SHA256"]
    old_paths = unit["actual_MD_paths_SHA256"]
    assert len(old_paths) == 575 and deps(old_md, old_obj) == set(old_paths)
    original, = [p for p in old_paths if Path(p).name == HEADER]
    assert old_paths[original] == q["original_probe_sha256"] and digest(original) == q["original_probe_sha256"]
    for path, pin in old_paths.items():
        assert digest(path) == pin, path
    new_header = D / HEADER
    assert digest(new_header) == q["candidate_probe_sha256"]
    link, = [list(a) for a in base["commands"] if "-shared" in a]
    assert set(base["all_linked_object_SHA256"]) == {v for v in link if v.endswith(".o")}

    OUT.mkdir()
    obj, md, pre = [OUT / n for n in ("worker.o", "worker.d", "worker.before.d")]
    cmd[1:1] = ["-iquote", str(D)]
    swap(cmd, old_obj, obj)
    swap(cmd, old_md, md)
    dry = list(cmd)
    swap(dry, "-MD", "-M")
    swap(dry, md, pre)
    k = dry.index("-o")
    del dry[k:k+2]
    dry.remove("-c")
    dry[1:1] = ["-MT", str(obj)]
    expected = (set(old_paths) - {original}) | {str(new_header)}
    start = time.monotonic()
    result = dict(passed=False, completed=False, GPU=0, Model=0, CPU_only=True, Native_dlopen=False,
                  SDK_execution=False, actual_CPU=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                  publication_commit=publication, claim_sha256=claim_sha)
    phase = "preprocessor"

    def run(label, argv, limit):
        left = q["inside_seconds"] - (time.monotonic() - start)
        if left <= 4:
            raise TimeoutError("CPU build deadline exceeded")
        child = subprocess.run(argv, capture_output=True, text=True, timeout=min(left-2, limit))
        (OUT / (label + ".log")).write_text(child.stdout + child.stderr)
        child.check_returncode()
        return child.stdout

    def exports(path, label):
        return {v[-1]:v[-2] for line in run(label, ["/usr/bin/nm","-D","--defined-only",str(path)],20).splitlines()
                if len(v:=line.split()) >= 3 and v[-2] in ("T","D","B","R")}

    def dynamic(path, label):
        raw = run(label, ["/usr/bin/readelf","-d",str(path)],15)
        assert "SYMBOLIC" in raw
        return re.findall(r"\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[(.*?)\]",raw)

    try:
        run("before-M", dry, 45)
        actual = deps(pre, obj)
        if actual != expected:
            result.update(missing=sorted(expected-actual)[:8], unexpected=sorted(actual-expected)[:8])
            raise ValueError("Kimi worker preprocessor dependency closure differs")
        phase = "compile"
        run("compile", cmd, 115)
        assert deps(md, obj) == expected
        for path, pin in old_paths.items():
            assert digest(path) == pin, path
        assert digest(new_header) == q["candidate_probe_sha256"]
        phase = "link"
        candidate = OUT / Path(base["native"]).name
        swap(link, old_obj, obj)
        swap(link, base["native"], candidate)
        run("link", link, 55)
        phase = "elf"
        old_symbols = exports(base["native"], "original-exports")
        assert exports(candidate, "candidate-exports") == old_symbols
        assert dynamic(candidate, "candidate-dynamic") == dynamic(base["native"], "original-dynamic")
        ldd = run("loader-relocations", ["/usr/bin/ldd","-r",str(candidate)],25)
        assert "not found" not in ldd and "undefined symbol" not in ldd
        objects = {p:digest(p) for p in link if p.endswith(".o")}
        assert len(objects) == 34
        assert {p:v for p,v in objects.items() if p != str(obj)} == {
            p:v for p,v in base["all_linked_object_SHA256"].items() if p != str(old_obj)}
        result.update(completed=True, passed=True, native=str(candidate), native_SHA256=digest(candidate),
                      all_linked_object_SHA256=objects, actual_dependency_library_SHA256=base["actual_dependency_library_SHA256"],
                      old_strong_exports_retained=base["old_strong_exports_retained"],old_CAPI_retained=base["old_CAPI_retained"],
                      new_CAPI=base["new_CAPI"], replaced_objects=1,reused_objects=33,dependency_paths=len(actual),
                      loader_relocations_checked=True, exact_prior_export_types_preserved=True)
    except Exception as error:
        result.update(failed_phase=phase, error=type(error).__name__)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic()-start
        with (OUT/"CPU-result.json").open("x",encoding="utf-8") as stream:
            json.dump(result,stream,indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
