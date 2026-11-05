"""Replace one audited DeepSeek C4 translation unit on a CPU-only step."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

from derive import ORIGINAL, derive

HERE = Path(__file__).resolve().parent
PARENT = Path("/refactor/candidates/Deep-C4-current-host-split-CPU-v1")
BASE = Path("/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/CPU-build.json")
OUTPUT = Path("/refactor/candidates/Deep-C4-compact-bank-CPU-private-state-v2")
ADDED = {"deep_c4_compact_bank_set_v1", "deep_c4_compact_bank_stats_v1"}


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def pinned(path, expected):
    if digest(path) != expected:
        raise ValueError(f"file differs: {path}")
    return json.loads(Path(path).read_bytes())


def argv_sha(argv):
    return sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()


def deps(path):
    _, sep, body = Path(path).read_text().replace("\\\n", " ").partition(": ")
    if not sep:
        raise ValueError("invalid compiler dependency file")
    return {Path(name).resolve(strict=True) for name in shlex.split(body)}


def swap(argv, old, new):
    if argv.count(str(old)) != 1:
        raise ValueError("compiler argument does not bind exactly one path")
    argv[argv.index(str(old))] = str(new)


def main():
    claim_sha = os.environ["DEEP_C4_COMPACT_CLAIM_SHA"]
    claim = pinned(HERE / "CPU_RESOURCE_CLAIM.json", claim_sha)
    assert claim["CPU_GO"] and claim["once"] and not claim["GPU_GO"] and not claim["Model_GO"]
    assert (os.environ["SLURM_JOB_ID"], socket.gethostname().split(".")[0]) == (claim["parent_job"], claim["node"])
    assert all(os.environ[k] == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    commit = os.environ["DEEP_C4_COMPACT_PUBLICATION"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    for name, value in claim["files_SHA256"].items():
        assert digest(HERE / name) == value
    prior = pinned(PARENT / "CPU-build.json", claim["parent_receipt_sha256"])
    baseline = Path(prior["library"])
    assert prior["completed"] and prior["actual_step"] == "910374.1458"
    assert digest(baseline) == prior["library_SHA256"] == claim["parent_dso_sha256"]
    base = pinned(BASE, claim["base_receipt_sha256"])
    for path, record in base["actual_DTK_bindings"].items():
        assert digest(path) == record["SHA256"]
    argv = {r["label"]: r["argv"] for r in prior["commands"]}
    original_argv = argv["compile-0"]
    assert argv_sha(original_argv) == claim["compile_argv_sha256"]
    link = list(argv["link"])
    assert argv_sha(link) == claim["link_argv_sha256"]
    objects = dict(base["all50_objects_SHA256"])
    for i, replaced in enumerate(prior["replaced"]):
        command = argv[f"compile-{i}"]
        objects.pop(replaced["parent"])
        objects[command[command.index("-o") + 1]] = replaced["SHA256"]
    assert len(objects) == 50 and set(objects) == {x for x in link if x.endswith(".o")}
    for path, pin in objects.items():
        assert digest(path) == pin, path
    source = Path(original_argv[original_argv.index("-c") + 1])
    obj = Path(original_argv[original_argv.index("-o") + 1])
    depfile = Path(original_argv[original_argv.index("-MF") + 1])
    assert digest(source) == ORIGINAL and digest(depfile) == claim["original_md_sha256"]
    code = derive(source.read_bytes())
    assert sha256(code).hexdigest() == claim["derived_source_sha256"]

    OUTPUT.mkdir()
    new_source, new_object, new_md, new_m = [OUTPUT / name for name in (source.name, obj.name, depfile.name, source.name + ".M")]
    new_source.write_bytes(code)
    command = [original_argv[0], "-iquote", str(PARENT), *original_argv[1:]]
    for old, new in ((source, new_source), (obj, new_object), (depfile, new_md)):
        swap(command, old, new)
    pre = list(command)
    swap(pre, "-MD", "-M")
    swap(pre, new_md, new_m)
    at = pre.index("-o")
    del pre[at:at+2]
    pre.remove("-c")
    expected = deps(depfile)
    expected.remove(source.resolve())
    expected.add(new_source.resolve())
    start = time.monotonic()
    state = dict(completed=False, actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                 publication_commit=commit, claim_sha256=claim_sha, GPU=False, Model=False, dlopen=False)
    phase = "preprocessor"

    def run(name, args):
        remaining = claim["inside_seconds"] - (time.monotonic() - start)
        if remaining <= 5:
            raise TimeoutError("CPU claim time exhausted")
        result = subprocess.run(["/usr/bin/timeout", "--kill-after=3s", f"{remaining-3:.2f}s", *args],
                                capture_output=True, text=True, timeout=remaining)
        (OUTPUT / f"{name}.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    def exports(path, label):
        result = run(label + "-exports", ["/usr/bin/nm", "-D", "--defined-only", str(path)])
        return {parts[-1] for row in result.splitlines()
                if len(parts := row.split()) >= 3 and parts[-2] in ("T", "D", "B", "R")}

    def dynamic(path, label):
        return re.findall(r"\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[(.*?)\]",
                          run(label + "-dynamic", ["/usr/bin/readelf", "-d", str(path)]))

    try:
        run("preprocessor", pre)
        got = deps(new_m)
        if got != expected:
            state["unexpected_dependency_count"] = len(got - expected)
            state["missing_dependency_count"] = len(expected - got)
            raise ValueError("C4 preprocessor dependency closure changed")
        phase = "compile"
        run("compile", command)
        assert deps(new_md) == expected
        phase = "link"
        native = OUTPUT / "libDeep_c4_compact_bank_v1.so"
        swap(link, obj, new_object)
        swap(link, baseline, native)
        run("link", link)
        phase = "ELF"
        old = exports(baseline, "baseline")
        new = exports(native, "candidate")
        assert old <= new and new - old == ADDED
        assert dynamic(baseline, "baseline") == dynamic(native, "candidate")
        state.update(completed=True, source_sha256=digest(new_source), object_sha256=digest(new_object),
                     new_exports=sorted(ADDED), retained_objects=49, replaced_objects=1,
                     dependency_paths=len(expected), library=str(native), library_SHA256=digest(native))
    except Exception as error:
        state.update(failed_phase=phase, error=type(error).__name__)
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - start
        (OUTPUT / "CPU-build.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
