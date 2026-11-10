"""Compile one CSA host unit and link the pinned 49-object parent closure."""
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shlex
import subprocess
import sys
import time

from derive import CSA_SHA, EXPORTS, HEADER, derive, restore

HERE = Path(__file__).resolve().parent
OUTPUT = Path("/build/artifacts")
PUBLIC = Path("/refactor/candidates/Deep-C4-history-bank-alias-CPU-v1/artifacts")

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def read(path, pin):
    assert digest(path) == pin, path
    return json.loads(Path(path).read_bytes())

def deps(path):
    body = path.read_text().replace("\\\n", " ").split(":", 1)[1]
    return {Path(x).resolve() for x in shlex.split(body)}

def change(argv, old, new):
    assert argv.count(str(old)) == 1, old
    argv[argv.index(str(old))] = str(new)

def main(claim_sha, deadline):
    assert os.environ["SLURM_JOB_ID"] == "1087656" and os.environ["SLURMD_NODENAME"] == "f12r3n12"
    assert all(os.environ[x] == "-1" for x in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    commit = os.environ["DEEP_HISTORY_PUBLICATION"]
    assert re.fullmatch("[0-9a-f]{40}", commit)
    seconds_left = deadline - time.time()
    assert 18 < seconds_left <= 300
    end = time.monotonic() + seconds_left - 18
    claim = read(HERE / "CPU_RESOURCE_CLAIM.json", claim_sha)
    assert claim["CPU_GO"] and not claim["GPU_GO"] and claim["once"]
    for name, pin in claim["files_SHA256"].items():
        assert digest(HERE / name) == pin, name
    proof = claim["original_receipts"]
    base = read(proof["link"]["path"], proof["link"]["SHA256"])
    baseline = Path(base["library"])
    assert digest(baseline) == base["library_SHA256"] == claim["baseline_Native_SHA256"]
    objects = base["all50_objects_SHA256"]
    assert len(objects) == 50
    for path, pin in objects.items():
        assert digest(path) == pin, path
    for path, record in base["actual_DTK_bindings"].items():
        assert digest(path) == record["SHA256"], path
    parent = read(proof["csa"]["path"], proof["csa"]["SHA256"])
    original = parent["new_compiled_units"][2]["compile_argv"]
    src = Path(original[original.index("-c") + 1])
    code = derive(src)
    assert sha256(restore(code)).hexdigest() == CSA_SHA
    OUTPUT.mkdir()
    state = dict(completed=False, GPU_initialized=False, Native_dlopen=False,
                 publication_commit=commit, claim_SHA256=claim_sha,
                 actual_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                 original_epoch=deadline-300, original_deadline=deadline,
                 replaced=[], reused=49, commands=[], elapsed_s=0)
    start = time.monotonic()

    def execute(label, argv):
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original CPU budget exhausted")
        state["commands"].append(dict(label=label, argv=argv))
        p = subprocess.run(argv, capture_output=True, text=True, timeout=min(110, remaining))
        (OUTPUT / (label+".log")).write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(f"{label} failed with code {p.returncode}")
        return p.stdout

    def exports(path):
        return {parts[-1] for line in execute(path.name+".symbols", ["nm", "-D", "--defined-only", str(path)]).splitlines()
                if len(parts := line.split()) >= 3 and parts[-2] in ("T", "D", "B", "R")}

    def needed(path):
        return re.findall(r"\(NEEDED\).*?\[(.*?)\]",
                          execute(path.name+".elf", ["readelf", "-d", str(path)]))

    try:
        (OUTPUT / HEADER).write_bytes((HERE / HEADER).read_bytes())
        old_object = Path(original[original.index("-o") + 1])
        old_md = Path(original[original.index("-MF") + 1])
        assert digest(old_md) == claim["original_MD_SHA256"][str(old_md)]
        assert str(old_object) in objects
        new_source = OUTPUT / src.name
        new_source.write_bytes(code.encode())
        new_object = OUTPUT / old_object.name
        new_md = OUTPUT / (src.name + ".d")
        new_m = OUTPUT / (src.name + ".M")
        argv = [*original[:1], "-iquote", str(src.parent), *original[1:]]
        for old, new in ((src, new_source), (old_object, new_object), (old_md, new_md)):
            change(argv, old, new)
        m = list(argv)
        change(m, "-MD", "-M")
        change(m, new_md, new_m)
        j = m.index("-o")
        del m[j:j+2]
        m.remove("-c")
        expected = deps(old_md)
        expected.remove(src.resolve())
        expected.update([new_source.resolve(), (OUTPUT / HEADER).resolve()])
        execute("deps", m)
        found = deps(new_m)
        if found != expected:
            state["dependency_difference"] = dict(added=sorted(map(str, found-expected)),
                                                   missing=sorted(map(str, expected-found)))
            raise ValueError("new -M differs from original closure plus alias header")
        execute("compile", argv)
        assert deps(new_md) == found
        state["replaced"].append(dict(parent=str(old_object), SHA256=digest(new_object),
            dependencies=len(found), source_SHA256=digest(new_source),
            M_SHA256=digest(new_m), MD_SHA256=digest(new_md), M_equals_MD=True,
            header_SHA256=digest(OUTPUT / HEADER)))
        link = list(base["link_argv"])
        change(link, old_object, new_object)
        lib = OUTPUT / "libDeep_c4_history_bank_alias_v1.so"
        change(link, baseline, lib)
        execute("link", link)
        old_exports = exports(baseline)
        new_exports = exports(lib)
        assert old_exports <= new_exports and new_exports-old_exports == set(EXPORTS)
        assert needed(baseline) == needed(lib)
        state.update(completed=True, library=str(PUBLIC/lib.name), library_SHA256=digest(lib),
            old_strong_exports=len(old_exports), new_strong_exports=sorted(new_exports-old_exports),
            dynamic_NEEDED_unchanged=True)
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        state["elapsed_s"] = time.monotonic()-start
        with (OUTPUT/"CPU-build.json").open("x", encoding="utf8") as stream:
            json.dump(state, stream, indent=2)

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
