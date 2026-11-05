"""Build the two C4 timing units under the original SDK without device execution."""
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shlex
import subprocess
import sys
import time

from derive import HEADER, derive

R = Path("/refactor")
HERE = Path(__file__).resolve().parent
OUTPUT = R / "candidates/Deep-C4-current-host-split-CPU-v1"


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


def main(claim_sha):
    assert os.environ["SLURM_JOB_ID"] == "910374" and os.environ["SLURMD_NODENAME"] == "a02r4n05"
    assert all(os.environ[x] == "-1" for x in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    commit = os.environ["DEEP_C4_SPLIT_PUBLICATION"]
    assert re.fullmatch("[0-9a-f]{40}", commit)
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
    csa = read(proof["csa"]["path"], proof["csa"]["SHA256"])
    index = read(proof["index"]["path"], proof["index"]["SHA256"])
    old_argv = [csa["new_compiled_units"][2]["compile_argv"], index["compiled"][0]["compile_argv"]]
    source_paths = [Path(x[x.index("-c")+1]) for x in old_argv]
    new_code = derive(*source_paths)
    OUTPUT.mkdir()
    state = dict(completed=False, GPU_initialized=False, Native_dlopen=False,
                 publication_commit=commit, actual_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                 replaced=[], reused=48, commands=[], elapsed_s=0)
    start = time.monotonic()

    def execute(label, argv):
        remaining = 250 - (time.monotonic() - start)
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
        link = list(base["link_argv"])
        plans = []
        for i, (src, code, original) in enumerate(zip(source_paths, new_code, old_argv)):
            old_object = Path(original[original.index("-o")+1])
            old_md = Path(original[original.index("-MF")+1])
            assert digest(old_md) == claim["original_MD_SHA256"][str(old_md)]
            assert str(old_object) in objects
            new_source = OUTPUT / src.name
            new_source.write_bytes(code.encode())
            new_object = OUTPUT / old_object.name
            new_md = OUTPUT / (src.name + ".d")
            new_m = OUTPUT / (src.name + ".M")
            argv = [*original[:1], "-iquote", str(src.parent), *original[1:]]
            for old,new in ((src,new_source),(old_object,new_object),(old_md,new_md)):
                change(argv,old,new)
            m = list(argv)
            change(m,"-MD","-M")
            change(m,new_md,new_m)
            j=m.index("-o")
            del m[j:j+2]
            m.remove("-c")
            expected = deps(old_md)
            expected.remove(src.resolve())
            expected.update([new_source.resolve(), (OUTPUT / HEADER).resolve()])
            execute(f"deps-{i}",m)
            found = deps(new_m)
            if found != expected:
                state["dependency_difference"] = dict(added=sorted(map(str,found-expected)),
                                                       missing=sorted(map(str,expected-found)))
                raise ValueError("new -M differs from original closure plus timing header")
            plans.append((argv,new_md,expected,new_object,old_object))
        for i,(argv,new_md,expected,new_object,old_object) in enumerate(plans):
            execute(f"compile-{i}",argv)
            assert deps(new_md) == expected
            change(link,old_object,new_object)
            state["replaced"].append(dict(parent=str(old_object),SHA256=digest(new_object),dependencies=len(expected)))
        lib = OUTPUT / "libDeep_c4_host_split_v1.so"
        change(link,baseline,lib)
        execute("link",link)
        old_exports=exports(baseline)
        new_exports=exports(lib)
        assert old_exports <= new_exports and "deep_c4_host_split_read_v1" in new_exports-old_exports
        assert needed(baseline)==needed(lib)
        state.update(completed=True,library=str(lib),library_SHA256=digest(lib),
                     old_strong_exports=len(old_exports),new_strong_exports=sorted(new_exports-old_exports),
                     dynamic_NEEDED_unchanged=True)
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        state["elapsed_s"]=time.monotonic()-start
        with (OUTPUT / "CPU-build.json").open("x",encoding="utf8") as stream:
            json.dump(state,stream,indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
